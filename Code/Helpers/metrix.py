# MetrixGen.py
import numpy as np

def gen(cmat):
  classes = cmat.shape[1]
  tp = np.zeros(classes)
  fp = np.zeros(classes)
  fn = np.zeros(classes)

  for x in range(classes):
    for y in range(classes):
      if(x == y):
        tp[x] = tp[x] + cmat[x][y]
      else:
        fn[x] = fn[x] + cmat[x][y]
        fp[y] = fp[y] + cmat[x][y]

  tn = np.sum(cmat) - (tp + fp)

  accuracy    = 100 * (tp + tn) / (tp + tn + fp + fn)
  MACC        = np.mean(accuracy)
  precision   = 100 * tp / (tp + fp)
  MPREC       = np.mean(precision)
  recall      = 100 * tp / (tp + fn)
  MREC        = np.mean(recall)
  IOU         = 100 * tp / (tp + fp + fn)
  MIOU        = np.mean(IOU)
  specificity = 100 * tn / (tn + fp)
  MSPEC       = np.mean(specificity)
  fonescore   = 2 * 1 / ( 1 / precision + 1 / recall)
  MF1         = np.mean(fonescore)

  met_score = MACC + MIOU + MF1

  print("\rMACC: %.2f MPREC: %.2f MREC: %.2f MIOU: %.2f MSPEC: %.2f MF1: %.2f"%(MACC,MPREC,MREC,MIOU,MSPEC,MF1))

  return met_score
