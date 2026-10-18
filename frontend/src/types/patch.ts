export interface Patch {
  id: string;
  prompt: string;
  embedding?: number[];
  created?: string;
  source?: string;
  action?: string;
  active?: boolean;
  status?: string;
  traffic_fraction?: number;
  scope?: string;
  match_regex?: string;
  match_substring?: string;
  rewrite_to?: string;
  // Metrics fields merged from backend
  score?: number;
  util_loss?: number;
  fp_rate?: number;
  p95_ov?: number;
}

export interface PatchMetrics {
  id: string;
  score: number;
  util_loss: number;
  fp_rate: number;
  p95_ov: number;
}

export interface LiveMetrics {
  safety_pass_rate: number;
  util_loss: number;
  fp_rate: number;
  latency: number;
}

export interface PatchSimilarity {
  patch: Patch;
  similarity: number;
}