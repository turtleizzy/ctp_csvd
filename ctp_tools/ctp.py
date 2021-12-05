import pydicom
from skimage.morphology import binary_dilation, binary_erosion
import cc3d
from scipy.optimize import curve_fit
import traceback
import SimpleITK as sitk
import numpy as np
import pandas as pd

def dcmLstToSitkImage(dcmLst, check_equal_spacing=True, sort_by='InstanceNumber', pixelID=sitk.sitkInt16):
    if len(dcmLst) == 0:
        raise ValueError('Empty dcmlst provided.')
    if callable(sort_by):
        dcmLst.sort(key=sort_by)
    else:
        dcmLst.sort(key=lambda x: x.get(sort_by))
    npArr = np.stack([i.pixel_array.astype('float') * i.get('RescaleSlope', 1) + i.get('RescaleIntercept', 0) for i in dcmLst])
    im = sitk.GetImageFromArray(npArr)
    im = sitk.Cast(im, pixelID)
    orientationW = np.array(dcmLst[0].ImageOrientationPatient[:3])
    orientationH = np.array(dcmLst[0].ImageOrientationPatient[3:])
    spacingW = dcmLst[0].PixelSpacing[0]
    spacingH = dcmLst[0].PixelSpacing[1]
    pos000 = np.array(dcmLst[0].ImagePositionPatient)

    if check_equal_spacing:
        vecDLst = np.stack([np.array(dcmLst[i+1].ImagePositionPatient) - np.array(dcmLst[i].ImagePositionPatient) for i in range(len(dcmLst)-1)])
        spacingDLst = np.linalg.norm(vecDLst, axis=1)

        # check equal spacing
        spacingDCounts = np.unique(np.round(spacingDLst * 100).astype('int'), return_counts=True)
        if len(spacingDCounts[0]) > 1:
            raise ValueError("Inequal spacing in depth direction. %s" % str(spacingDCounts))
        spacingD = spacingDLst[0]

        # check colinearity
        vecDNorm = vecDLst / spacingDLst[:, np.newaxis]
        orientationD = vecDNorm[0]

        if np.any(np.abs(np.dot(vecDNorm, orientationD) - 1) > 1e-4):
            raise ValueError("Depth direction is not linearly stacked. %s" % str(np.dot(vecDNorm, orientationD)))

    else:
        pos001 = np.array(dcmLst[1].ImagePositionPatient)
        spacingD = np.linalg.norm(pos001-pos000)
        orientationD = (pos001 - pos000) / spacingD

    direction = np.vstack([orientationW, orientationH, orientationD])
    spacing = np.array([spacingW, spacingH, spacingD])

    im.SetOrigin(pos000.tolist())
    im.SetDirection(direction.T.ravel().tolist()) # transpose is necessary because directions are raveled as [w_x, w_y, w_z, h_x, h_y, h_z, ...]
    im.SetSpacing(spacing.tolist())
    
    return im

def readCTPSeries(dcmList, refTime=None, fixedInterval=None, debug=False):
    ds = []
    for i in dcmList:
        ds.append(pydicom.read_file(i))
    df = []
    for i in ds:
        df.append({
            'acqTime': pd.to_datetime(f'{i.AcquisitionDate} {i.AcquisitionTime}'),
            'contentTime': pd.to_datetime(f'{i.ContentDate} {i.ContentTime}'),            
            'instanceNumber': i.InstanceNumber,
            'acqNumber': i.AcquisitionNumber,
            'dcmObj': i
        })
    df = pd.DataFrame(df)
    df.sort_values(by='instanceNumber', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if refTime is not None:
        if refTime == 'AcquisitionTime':
            df['refTime'] = df.acqTime
        elif refTime == 'ContentTime':
            df['refTime'] = df.contentTime
        else:
            raise NotImplementedError(f"{refTime} as refTime is not supported.")
        df['timeDiff'] = df.refTime.diff()
        df['timeDiff'].iloc[0] = pd.to_timedelta(0)
        locs = np.concatenate([[0], df[df.timeDiff > pd.to_timedelta('0.5s')].index])
    else:
        if fixedInterval is not None:
            df['refTime'] = np.floor((df.instanceNumber - 1).astype('float') / fixedInterval)
        else:
            df['refTime'] = df.acqNumber
        df['timeDiff'] = df.refTime.diff()
        df['timeDiff'].iloc[0] = 0
        locs = np.concatenate([[0], df[df.timeDiff > 0].index])

    times = []
    images = []
    for tp in range(len(locs)-1):
        st = locs[tp]
        ed = locs[tp+1]
        im = dcmLstToSitkImage(df.iloc[st:ed].dcmObj.tolist())
        times.append(df.iloc[st].refTime)
        images.append(im)
    if refTime is not None:
        times = np.array([(i - times[0]).total_seconds() for i in times])
    else:
        times = np.array([(i - times[0]) for i in times])
    if debug:
        return images, times, df, locs
    else:
        #im = sitk.Compose(images)
        return images, times

def getBrainmask(im):
    im = sitk.DiscreteGaussian(im)
    boneLabelMap = sitk.ConnectedComponent(im > 300)

    labelStat = sitk.LabelShapeStatisticsImageFilter()
    labelStat.Execute(boneLabelMap)

    labelLst = [(i, im.TransformPhysicalPointToIndex(labelStat.GetCentroid(i)), labelStat.GetPhysicalSize(i)) for i in
                labelStat.GetLabels()]

    seed = max(labelLst, key=lambda x: x[2])[1]

    feature_img = sitk.GradientMagnitudeRecursiveGaussian(im, sigma=.5)
    speed_img = sitk.BoundedReciprocal(feature_img)
    fm_filter = sitk.FastMarchingBaseImageFilter()
    fm_filter.SetTrialPoints([seed])
    fm_filter.SetStoppingValue(400)
    fm_img = fm_filter.Execute(speed_img)

    out=fm_img < 400

    return out

def getCTC(imLst, brainMask, huThres=150, ratioThres=0.05):
    imNp = np.stack([sitk.GetArrayFromImage(im) for im in imLst])
    brainMaskNp = sitk.GetArrayFromImage(brainMask)
    totContrastAgentVal = np.array(
        [imNp[t][((imNp[t] > huThres) & brainMaskNp).astype('bool')].sum() for t in range(imNp.shape[0])]
    ) + 1
    totContrastAgentVal = (totContrastAgentVal - totContrastAgentVal[0]) / totContrastAgentVal[0]
    cands = np.where(totContrastAgentVal > ratioThres)[0]
    if cands.shape[0] > 0:
        s0Idx = cands[0]
        s0ImNp = imNp[:s0Idx].mean(axis=0)
        imNp = imNp - s0ImNp
        imNp[imNp<0] = 0
        return imNp, s0Idx
    else:
        return None, None
    
def gv(t, t0, alpha, beta):
    return np.maximum(0, t-t0)**alpha * np.exp(-t/beta) * (np.sign(t - t0) + 1) / 2

def fit_gv(tIdx, curve, sigma=None):
    return curve_fit(gv, tIdx, curve, bounds=([0, 0.1, 0.1], [20, 8, 8]), sigma=sigma)

def targetF(x_obs, y_obs):
    def gv_target(t0, alpha, beta):
        return (gv(x_obs, t0, alpha, beta) - y_obs) ** 2
    return gv_target

def majority(arr, ignore=None):
    labs, cnts = np.unique(arr.ravel(), return_counts=True)
    if ignore is not None:
        mask = labs.isin(ignore)
        cnts = cnts[~mask]
        labs = labs[~mask]
        return labs[np.argmax(cnts)]
    else:
        return labs[np.argmax(cnts)]
    
def runTTP(imCTC, tIdx, s0Idx, brainMask, outsideValue=-1):
    tIdxFil = tIdx[s0Idx:] - tIdx[s0Idx]
    imCTCFil = imCTC[s0Idx:]
    TTP = tIdxFil[imCTCFil.argmax(axis=0)]
    TTP[sitk.GetArrayFromImage(brainMask) == 0] = outsideValue
    return TTP

def runAIF(imCTC, tIdx, brainMask, TTP, 
           roi=None, 
           TTPThres=8, AUCThres=1000, 
           dilateRadius=5, erodeRadius=2,
           candidateVolThres = 2000
          ):
    brainMaskNp = sitk.GetArrayFromImage(brainMask)
    
    aifCand = (imCTC.sum(0) * brainMaskNp > AUCThres) * (TTP < TTPThres) * (TTP > 0)
    
    if roi is not None:
        aifCandRoi = np.zeros_like(aifCand)
        aifCandRoi[roi[2]:roi[5], roi[1]:roi[4], roi[0]:roi[3]] =  aifCand[roi[2]:roi[5], roi[1]:roi[4], roi[0]:roi[3]]
        aifCand = aifCandRoi
    
    aifCand = binary_erosion(
        binary_dilation(aifCand, np.ones([1, dilateRadius, dilateRadius], bool)), 
        np.ones([1, erodeRadius, erodeRadius], bool))
    
    aifCand, nComp = cc3d.connected_components(aifCand, return_N=True)

    cands = []

    for idx in range(1, nComp+1):
        curCand = aifCand == idx
        vol = curCand.sum()
        curve = imCTC[:, curCand]
        curveMean = curve.mean(axis=1)
        peakEndDiff = np.max(curveMean) - np.mean(curveMean[-3:])
        if vol * np.prod(brainMask.GetSpacing()) > candidateVolThres:
            continue
        try:
            popts, _ = fit_gv(tIdx, curveMean)
        except:
            traceback.print_exc()
            continue
        err = np.sqrt(np.sum((curve - gv(tIdx, *popts)[:, np.newaxis]) ** 2, axis=1))
        cands.append({
            'idx': idx,
            'vol': vol * np.prod(brainMask.GetSpacing()),
            'popts': popts,
            'meanErr': err.mean(),
            'maxErr': err.max(),
            'peakEndDiff': peakEndDiff
        })

    cands = pd.DataFrame(cands)
    cands['score'] = cands.vol * cands.peakEndDiff / cands.meanErr 
    bestCand = np.argmax(cands.score)
    aifSegIdx = cands.iloc[bestCand].idx
    aifProps = cands.iloc[bestCand].popts
    cands['chosen'] = 0
    cands['chosen'].iloc[bestCand] = 1
    return aifProps, aifCand==aifSegIdx, cands

def viewAIFDiagnosticImage(im, tIdx, imCTC, s0Idx, aifProps, aifCandSeg, window=(100, 1200)):
    import matplotlib.pyplot as plt
    
    majSlice = majority(np.where(aifCandSeg)[0])
    majIm = im[(imCTC.shape[0] - s0Idx)//2 + s0Idx][:, :, int(majSlice)]
    majIm = sitk.IntensityWindowing(majIm, window[0]-window[1]/2, window[0]+window[1]/2, 0, 255)
    majLab = sitk.GetImageFromArray(aifCandSeg[majSlice].astype('uint8'))
    majLab.CopyInformation(majIm)
    majIm = sitk.GetArrayFromImage(sitk.LabelOverlay(majIm, majLab))
    ax = plt.subplot(211)
    ax.imshow(majIm.astype('uint8'))
    ax2 = plt.subplot(212)
    ax2.plot(tIdx, imCTC[:, aifCandSeg].mean(axis=1))
    ax2.plot(np.linspace(0, max(tIdx)*1.5, 1000), gv(np.linspace(0, max(tIdx)*1.5, 1000), *aifProps))
    
def viewImDiagnostic(imLst, window=(200, 500), cols=6):
    import matplotlib.pyplot as plt

    rows = int(np.ceil(float(len(imLst)) / cols))
    for idx, i in enumerate(imLst):
        ax = plt.subplot(rows, cols, idx+1)
        imSlice = sitk.IntensityWindowing(
            sitk.MaximumProjection(i[:, :, i.GetDepth()//3:i.GetDepth()*2//3], 2)[:, :, 0], 
            window[0]-window[1]/2, window[0]+window[1]/2, 0, 255)
        imSlice = sitk.GetArrayFromImage(imSlice).astype('uint8')
        ax.axis('off')
        ax.imshow(imSlice, cmap=plt.cm.Greys_r)

def runDeconv(imCTC, tIdx, brainMask, aifProps, cSVDThres=0.1, outputMethod='aggregate', outsideValue=-1):
    brainMaskNp = sitk.GetArrayFromImage(brainMask)
    # obsAifVal = imCTC[:, aifCand == aifSegIdx].mean(axis=1)
    estimAifVal = gv(tIdx, *aifProps)
    aifVal = estimAifVal

    CBV = np.sum(imCTC, axis=0) / np.sum(aifVal)
    CBV[brainMaskNp == 0] = outsideValue
    
#     colG = np.zeros(2 * tIdx.shape[0])
#     colG[0] = aifVal[0]
#     colG[tIdx.shape[0]-1] = (aifVal[tIdx.shape[0]-2] + 4 * aifVal[tIdx.shape[0]-1]) / 6
#     colG[tIdx.shape[0]] = aifVal[tIdx.shape[0]-1] / 6
#     for k in range(1, tIdx.shape[0]-1):
#         colG[k] = (aifVal[k-1] + 4 * aifVal[k] + aifVal[k+1]) / 6

#     rowG = np.zeros(2 * tIdx.shape[0])
#     rowG[0] = colG[0]
#     for k in range(1, 2 * tIdx.shape[0]):
#         rowG[k] = colG[2*tIdx.shape[0] - k]

#     G = toeplitz(colG, rowG)

    cmat = np.zeros([tIdx.shape[0], tIdx.shape[0]])
    B = np.zeros([tIdx.shape[0], tIdx.shape[0]])
    for i in range(tIdx.shape[0]):
        for j in range(tIdx.shape[0]):
            if i == j:
                cmat[i, j] = aifVal[0]
            elif i > j:
                cmat[i, j] = aifVal[(i-j)]
            else:
                B[i, j] = aifVal[tIdx.shape[0] - (j-i)]
    G = np.vstack([np.hstack([cmat, B]),np.hstack([B, cmat])])

    U, S, V = np.linalg.svd(G)
    thres = cSVDThres * np.max(S)
    filteredS = 1 / S
    filteredS[S < thres] = 0
    Ginv = V @ np.diag(filteredS) @ U.T

    imCTC_pad = np.pad(imCTC, [(0, imCTC.shape[0]),] + [(0, 0)]*3)
    CBF_interm = np.abs(np.einsum('ab, bcde->acde', Ginv, imCTC_pad))
    if outputMethod == 'max':
        CBF = CBF_interm.max(axis=0)
    elif outputMethod == 'aggregate':
        CBF = np.abs(CBF_interm[2:-1] - 2 * CBF_interm[1:-2] + CBF_interm[:-3]).sum(0)
        CBF /= CBF_interm.shape[0]
    else:
        raise NotImplementedError(f"outputMethod {outputMethod} is not supported.")
    CBF[brainMaskNp == 0] = outsideValue
    MTT = CBV / CBF
    MTT[brainMaskNp == 0] = outsideValue
    
    return MTT, CBV, CBF

