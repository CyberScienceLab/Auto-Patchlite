**Metrics and denominators (oracle-defined)**
ASRpolicy = (# harmful attempts judged successful by oracle policy) / Nharm.
ASRcampaign_policy = (# successful harmful attempts across campaign windows) / Ncampaign_attempt.
VRharm-harness = (# harmful traces where harness-level validator fires) / Nharm.
Nseed = number of random seeds (independent experimental initializations).
Nattempt = number of total model attempts in a run slice.
Ncampaign_attempt = number of attempts across all campaign intervals.
Nharm = number of harmful prompts/attempts evaluated.
Nben = number of benign prompts/attempts evaluated.
Refusal rate = (# refusals across harmful+benign) / (Nharm + Nben).
Benign correctness = (# benign outputs accepted and non-harmful in oracle) / Nben.
All reported rates are oracle-defined and not audit-validated ground truth.
