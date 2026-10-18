import gymnasium as gym
from gymnasium import spaces
import numpy as np

class JailbreakDefenseEnv(gym.Env):
    def __init__(self, dataset, difficulty='easy'):
        super(JailbreakDefenseEnv, self).__init__()
        self.np_random = np.random.default_rng()
        self.dataset = dataset
        self.difficulty = difficulty
        self.unsafe_indices = []
        self.safe_indices = []
        self._update_filtered_indices()
        self.num_samples = len(self.filtered_indices)

        self.prompt_dim = len(self.dataset[0]['prompt_embedding'])
        self.extra_features = [
            'prompt_system_role_token_hit',
            'target_contains_refusal',
            'target_response_length',
            'target_char_entropy'
        ]
        self.extra_dim = len(self.extra_features)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.prompt_dim + 5 + self.extra_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(5)
        self.current_index = 0

    def filter_dataset(self):
        return [i for i, ex in enumerate(self.dataset) if ex['difficulty'] == self.difficulty]

    def _update_filtered_indices(self):
        self.filtered_indices = self.filter_dataset()
        self.unsafe_indices = []
        self.safe_indices = []
        for idx in self.filtered_indices:
            ex = self.dataset[idx]
            base_unsafe_label = ex['target_contains_policy_violation']
            high_jb_risk = ex['prompt_jailbreak_likelihood'] > 0.7
            high_policy_risk = ex['policy_violation_logit'] > 0.7
            guardrail_flagged = bool(ex['wrapper_flagged'])
            if (base_unsafe_label or (high_jb_risk and high_policy_risk) or guardrail_flagged):
                self.unsafe_indices.append(idx)
            else:
                self.safe_indices.append(idx)

    def reset(self, seed=None, options=None, forced_index=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if forced_index is not None:
            self.current_index = forced_index
        else:
            if self.unsafe_indices and self.safe_indices:
                chosen_list = self.unsafe_indices if self.np_random.uniform() < 0.5 else self.safe_indices
                self.current_index = int(self.np_random.choice(chosen_list))
            else:
                self.current_index = int(self.np_random.choice(self.filtered_indices))

        obs = self._get_obs(self.current_index)
        return obs, {}


    def _get_obs(self, idx):
        ex = self.dataset[int(idx)]
        prompt_emb = np.array(ex['prompt_embedding'], dtype=np.float32)
        toxicity = np.array([ex['prompt_contains_profanity']], dtype=np.float32)
        jailbreak_heuristic = np.array([ex['prompt_jailbreak_likelihood']], dtype=np.float32)
        policy_logit = np.array([ex['policy_violation_logit']], dtype=np.float32)
        latency = np.array([ex['simulated_latency_ms']], dtype=np.float32)
        guardrail_flag = np.array([ex['wrapper_flagged']], dtype=np.float32)
        latency_norm = latency / 1000.0

        extra = []
        for feat in self.extra_features:
            val = ex.get(feat, 0)
            if feat == 'target_response_length':
                val = float(val) / 100.0
            elif feat == 'target_char_entropy':
                val = float(val) / 8.0
            extra.append(float(val))
        extra = np.array(extra, dtype=np.float32)

        return np.concatenate([
            prompt_emb,
            toxicity, jailbreak_heuristic,
            policy_logit, latency_norm, guardrail_flag,
            extra
        ])

    def step(self, action):
        idx = self.current_index
        ex = self.dataset[int(idx)]

        # === Label and Risk Info ===
        base_unsafe_label = ex['target_contains_policy_violation']
        high_jb_risk = ex['prompt_jailbreak_likelihood'] > 0.7
        high_policy_risk = ex['policy_violation_logit'] > 0.7
        guardrail_flagged = bool(ex['wrapper_flagged'])
        task_intent_preserved = ex.get('rewrite_preserves_intent', True)
        latency_ms = ex['simulated_latency_ms']

        system_role_token_hit = bool(ex.get('prompt_system_role_token_hit', False))
        profanity = bool(ex.get('prompt_contains_profanity', False))

        high_risk = any([high_jb_risk, high_policy_risk, guardrail_flagged, system_role_token_hit, profanity])

        reward = 0.0

        # === Safe ===
        if base_unsafe_label == 0:
            if action == 0:  # ALLOW
                reward += 2.3  # Strongly reward correct allow
                #if high_risk:
                #    reward -= 0.5  # Caution penalty
            elif action == 1:  # BLOCK
                reward -= 1.3  # Strong false positive penalty
            elif action == 2:  # REWRITE
                #reward -= 0.3
                reward -= 0.5  # Penalty for unnecessary rewrite
            elif action == 3:  # ASK
                #reward -= 0.2
                reward -= 0.25
            elif action == 4:  # ESCALATE
                #reward -= 0.2
                reward -= 0.25

        # === Unsafe ===
        elif base_unsafe_label == 1:
            if action == 0:  # ALLOW
                reward -= 1.5  # Harshly penalize unsafe allows
                if high_risk:
                    reward -= 0.1
            elif action == 1:  # BLOCK
                reward += 2.3
                if high_risk:
                    reward += 0.3
            elif action == 2:  # REWRITE
                reward += 2.2 if task_intent_preserved else 1.3
            elif action == 3:  # ASK
                reward += 1.2  # Up from 1.2
                if high_risk:
                    reward += 0.3
            elif action == 4:  # ESCALATE
                #elif action == 4:  # ESCALATE
                reward += 1.2  # Safe, but not optimal
                if high_risk:
                    reward += 0.3  # Slight bonus for caution on high risk

        # === Unsure ===
        elif base_unsafe_label == -1:
            if action == 0:  # ALLOW
                reward -= 0.8
                if high_risk:
                    reward -= 0.2
            elif action == 1:  # BLOCK
                reward += 0.5
            elif action == 2:  # REWRITE
                reward += 1.5 if task_intent_preserved else 0.7 
            elif action == 3:  # ASK
                #reward += 0.9
                reward += 0.8
                if high_risk:
                    reward += 0.4
            elif action == 4:  # ESCALATE
                #reward += 0.9  # Cautious fallback
                reward += 0.8
                if high_risk:
                    reward += 0.4

        # === Latency penalty ===
        excess_latency = max(0.0, (latency_ms - 200.0) / 1000.0)
        reward -= 0.01 * excess_latency

        # === Feature Modifiers ===
        feature_penalty = 0.0
        if guardrail_flagged:
            feature_penalty += 0.01
            #if action in [3, 4]:
            #    reward += 0.2
        if high_jb_risk:
            feature_penalty += 0.01
        if high_policy_risk:
            feature_penalty += 0.01
        if system_role_token_hit:
            feature_penalty += 0.01
            #if action in [2,3]:
            #    reward += 0.2
        if profanity:
            feature_penalty += 0.05
            #if action == 4:
            #    reward += 0.2

        # Discourage ALLOW when flagged
        if action == 0 and base_unsafe_label == 1 or base_unsafe_label == -1:
            reward -= feature_penalty
        else:
            reward += feature_penalty

        # === Small entropy bonus (encourage exploring non-block actions) ===
        if action in [2, 3, 4]:  # REWRITE, ASK, ESCALATE
            reward += 0.1  # Encourage using REWRITE, ASK, ESCALATE

        # Optional: noise to encourage stochastic exploration
        reward += self.np_random.normal(0, 0.01)


        # === Episode ends each step ===
        terminated = True
        truncated = False
        obs, _ = self.reset()

        info = {
            "prompt": ex.get("prompt"),
            "true_label": base_unsafe_label,
            "action_taken": int(action),
            "latency_ms": latency_ms,
            "guardrail_flagged": guardrail_flagged,
            "high_jb_risk": high_jb_risk,
            "high_policy_risk": high_policy_risk,
            "system_role_token_hit": system_role_token_hit,
            "profanity": profanity,
        }

        return obs, reward, terminated, truncated, info
