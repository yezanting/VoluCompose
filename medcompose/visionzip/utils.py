def cal_compose_rate(mask):
  """Realized retention fraction rho of a boolean keep mask (kept / total)."""
  return mask.sum()/(mask.shape[0])