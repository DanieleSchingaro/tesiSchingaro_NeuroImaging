#src/evaluation/metrics.py
"""
Metriche per la valutazione delle MRI cerebrali sintetiche (HC) generate dall'LDM.
Allineato a NV-Generate-CTMR (MAISI): tutte le metriche vengono da
'generative.metrics' (monai-generative), usata da Maisi stessa.
    
    FIDMetric        -> vuole FEATURE gia' estratte: fid(synth_feat, real_feat).
                      Replichiamo l'approccio 2.5D di MAISI estraendo feature 2D
                      dalle slice dei 3 piani ortogonali (XY/YZ/ZX) con un
                      estrattore Inception, poi FID per piano + media.
    MultiScaleSSIMMetric -> diversita' intra-set (mode collapse).
    MMDMetric        -> mmd(y, y_pred) direttamente sui volumi 3D.
 
Estrattore di feature per il FID: InceptionV3 (pesi ImageNet) via torchvision.
MAISI usa RadImageNet ResNet50, che pero' nel nostro ambiente non si carica
(mismatch strutturale delle chiavi backbone.*). La sostituzione con Inception
e' supportata da Woodland et al. (MICCAI 2024): gli estrattori ImageNet sono
piu' consistenti e allineati al giudizio umano rispetto a RadImageNet nel
medical imaging. La logica 2.5D resta identica a quella di MAISI.
"""

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import inception_v3, Inception_V3_Weights
from generative.metrics import FIDMetric
from generative.metrics import MultiScaleSSIMMetric
from generative.metrics import MMDMetric

#Estrattore di feature 2D: InceptionV3
class InceptionFeatureExtractor(nn.Module):
    """
    Wrapper su InceptionV3 (pesi ImageNet) che restituisce il vettore di feature
    a 2048 dimensioni (output del global average pooling, prima della fc finale).
    Le slice 2D vengono: replicate a 3 canali, ridimensionate a 299x299
    (input nativo di Inception), normalizzate con mean/std di ImageNet.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        weights=Inception_V3_Weights.IMAGENET1K_V1
        net=inception_v3(weights=weights, aux_logits=True)
        net.fc=nn.Identity()
        net.eval()
        self.net=net.to(device)
        self.device=device 
        #normalizzazione ImageNet
        self.mean=torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        self.std=torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    
    @torch.no_grad()
    def forward(self, slices_2d:torch.Tensor)->torch.Tensor:
        """
        slices_2d: [N,H,W] in [0,1] (scala di grigi)
        Ritorna: [N, 2048] feature
        """
        x=slices_2d.unsqueeze(1) #[N,1,H,W]
        x=x.repeat(1,3,1,1) #scala di grigi
        x=F.interpolate(x, size=(299,299), mode="bilinear", align_corners=False)
        x=(x-self.mean)/self.std #normalizzazione ImageNet
        feat=self.net(x)
        return feat

#Iterazione slice 2D sui tre piani (con drop delle slice quando vuote)
def _iter_plane_slices(volume:torch.Tensor, plane:str, drop_empty:bool=True,
                       empty_frac:float=0.01):
    """
    Itera le slice 2D di un volume 3D [H,W,D] lungo il piano scelto.
      plane "xy": lungo D ; "yz": lungo H ; "zx": lungo W.
    Se drop_empty, scarta le slice quasi nere (cervello skull-stripped):
    quelle ai bordi sono quasi tutte sfondo e identiche tra volumi, gonfierebbero
    artificialmente la similarita'.
    """
    if plane=="xy":
        n=volume.shape[2]; getter=lambda i: volume[:,:,i]
    elif plane=="yz":
        n=volume.shape[0]; getter=lambda i: volume[i,:,:]
    elif plane=="zx":
        n=volume.shape[1]; getter=lambda i: volume[:,i,:]
    else:
        raise ValueError(f"piano sconosciuto: {plane}")
    
    for i in range(n):
        s=getter(i)
        if drop_empty:
            frac=(s>0.02).float().mean().item()
            if frac<empty_frac:
                return
        yield s 

@torch.no_grad()
def _extract_features_2p5d(volumes, extractor, plane, device,
                           drop_empty=True, batch_size=32):
    """
    Estrae e impila le feature 2D di tutte le slice (lungo 'plane') di tutti i volumi.
    Restituisce un tensore [N_slices_totali, 2048].
    """
    feats=[]
    buf=[]
    def flush():
        if not buf:
            return
        batch=torch.stack(buf).to(device) #[N,H,W]
        f=extractor(batch) #[N,2048]
        feats.append(f.cpu())
        buf.clear()
    
    for vol in volumes:
        vol=vol.to(device)
        for s in _iter_plane_slices(vol, plane, drop_empty=drop_empty):
            buf.append(s)
            if len(buf)>=batch_size:
                flush()
    
    flush()
    return torch.cat(feats, dim=0) if feats else torch.empty(0,2048)


#FID 2.5
@torch.no_grad()
def compute_fid_2p5d(real_volumes, synth_volumes, device="cuda",
                     drop_empty=True, batch_size=32, verbose=True):
    """
    FID 2.5D tra reali e sintetiche sui 3 piani ortogonali + media.
    Estrae feature 2D con Inception (logica MAISI), poi usa generative.FIDMetric
    che vuole feature già estratte.

    real_volumes/synth_volumes: liste di tensori 3D [H,W,D] in [0,1].
    Returns: dict con fid_xy, fid_yz, fid_zx, fid_avg.
    """
    extractor=InceptionFeatureExtractor(device=device)
    fid_metric=FIDMetric()
    results={}

    for plane in ["xy", "yz", "zx"]:
        real_feat=_extract_features_2p5d(real_volumes, extractor, plane, device,
                                         drop_empty=drop_empty, batch_size=batch_size)
        synth_feat=_extract_features_2p5d(synth_volumes, extractor, plane, device,
                                          drop_empty=drop_empty, batch_size=batch_size)
        
        score=float(fid_metric(synth_feat, real_feat).item())
        results[f"fid_{plane}"]=score
        if verbose:
            print(f"FID {plane.upper()}: {score:.3f}   "
                  f"(slice reali={real_feat_shape[0]}, sint={synth_feat.shape[0]})")
        if device != "cpu":
            torch.cuda.empty_cache()
    
    results["fid_avg"]=(results["fid_xy"] + results["fid_yz"] + results["fid_zx"])/3.0
    if verbose:
        print(f"FID medio (3 piani): {results['fid_avg']:.3f}")
    return results

#MSSIM
@torch.no_grad()
def compute_msssim_diversity(volumes, device="cuda", n_pairs=200, seed=42, verbose=True):
    """
    Diversita' di un insieme di volumi = MS-SSIM medio tra COPPIE casuali di
    volumi diversi.
      - MS-SSIM ALTO tra coppie -> volumi simili tra loro -> poca varieta'
        (possibile mode collapse).
      - MS-SSIM BASSO -> buona varieta'.
    Va confrontato tra set: si calcola su sintetiche E su reali, poi si
    confrontano i due valori (le sintetiche dovrebbero avere diversita' simile
    alle reali, non molto piu' alta).
 
    NB sui parametri MS-SSIM 3D: i 5 pesi di default implicano 5 livelli di
    downsampling; con volumi 256^3 e kernel 11 i livelli alti scendono sotto la
    dimensione del kernel -> errore. Usiamo 3 pesi e kernel 7, adatti al 3D.
    """
    msssim=MultiScaleSSIMMetric(
        spatial_dims=3,
        data_range=1.0,
        kernel_size=7,
        weights=(0.0448, 0.2856, 0.3001),
    )

    n=len(volumes)
    if n<2:
        return float("nan")
    
    rng=np.random.default_rng(seed)
    scores=[]
    for _ in range(n_pairs):
        i,j=rng.choice(n, size=2, replace=False)
        a=volumes[i].to(device).unsqueeze(0).unsqueeze(0) #[1,1,H,W,D]
        b=volums[j].to(device).unsqueeze(0).unsqueeze(0)
        val=msssim(a,b)
        scores.append(float(val.mean().item()))
        if device != "cpu":
            torch.cuda.empty_cache()
    
    mean_scores=float(np.mean(scores))
    if verbose:
        print(f"MS-SSIM medio su {n_pairs} coppie: {mean_scores:.4f}")
    return mean_scores

#MMD: Maximum Mean Discrepancy
@torch.no_grad()
def compute_mmmd(real_volumes, synth_volumes, device="cuda", verbose=True):
    """
    MMD tra reali e sintetiche con generative.MMDMetric: mmd(y, y_pred) accetta
    tensori 3D [B,C,W,H,D]. Valori piu' bassi = distribuzioni piu' vicine.
    Accoppiamo i volumi disponibili (troncando al minimo comune) e mediamo.
 
    NB: l'MMD qui e' calcolata sui volumi grezzi. La MMDMetric supporta un
    y_transform/y_pred_transform opzionale (es. filtro gaussiano o estrattore di
    feature); lo lasciamo a None (identita') per semplicita' e coerenza.
    """
    mmd=MMDMetric()
    n=min(len(real_volumes), len(synth_volumes))
    scores=[]
    for k in range(n):
        y=real_volumes[k].to(device).unsqueeze(0).unsqueeze(0)
        y_pred=synth_volumes[k].to(device).unsqueeze(0).unsqueeze(0)
        val=mmd(y, y_pred)
        scores.append(float(val.mean().item()))
        if device != "cuda"
        torch.cuda.empty_cache()
    
    mean_scores=float(np.mean(scores))
    if verbose:
        print(f"MMD medio: {mean_scores:.6f}")
    return mean_scores