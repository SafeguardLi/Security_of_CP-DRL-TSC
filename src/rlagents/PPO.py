from src.attrlagent import AttRLAgent
import torch
import torch.nn.functional as F
import numpy as np
import os
import random
from src.helper_funcs import check_and_make_dir, write_line_to_csv, get_fp

class PPOAgent(AttRLAgent):
    def __init__(self, networks, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma, rl_stats, mode,
                 updates, main_args, tsc_id):
        super().__init__(networks, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma, rl_stats,
                         mode, updates)
        self.main_args = main_args
        self.tsc_id = tsc_id
        self.noise = True
        self.global_critic = main_args.global_critic 
        self.max_attack_scale = main_args.max_attack_scale
        
        self.gae_lambda = 0.995
        self._last_applied_beta = None
        
        # --- Hyperparameters ---
        self.eta = 0.5 
        # Removed jsma_lambda and delay_lambda since we only have one objective now
        
        self.epsilon_PPO = 0.2
        self.MINIBATCH_SIZE = n_batch if n_batch > 0 else 128
        self.total_action_cnt = 0
        self.same_action_cnt = 0

    def rule_based_attack_guide(self,state, rule='default',next_phase=None):
        if rule == 'default':
            return 0 
        elif rule == 'phase':
            if next_phase==None:
                print("ERROR. Rule==phase while next_phase is None")
                return 0
            action_pair = {0: 1, 1: 1, 2: 1, 3: 0}
            return action_pair.get(next_phase, 0)
        elif rule == 'always0':
            return 0
        elif rule == 'always1':
            return 1
        elif rule == 'random':
            return random.randint(0,1)
        elif rule == 'minPressure':
            phase_slices = [(36, 39), (39, 42), (42, 43), (43, 46)]
            
            # Extract the specific slice for the current 'next_phase'
            start, end = phase_slices[next_phase]
            
            # Calculate pressure for the current phase
            current_pressure = sum(state[start:end])
            
            # Calculate the individual pressures for all OTHER phases
            other_phase_pressures = []
            for i, (s, e) in enumerate(phase_slices):
                if i != next_phase:
                    other_phase_pressures.append(sum(state[s:e]))
            
            # Find the maximum pressure among the other phases
            max_other_pressure = max(other_phase_pressures)
            
            print("next phase idx:", next_phase, 
                  "current_pressure:", current_pressure, 
                  "max_other_pressure:", max_other_pressure)
            
            # Return 0 (Switch) if current pressure is greater than or equal to the 
            # highest pressure found in the rest of the phases, else 1 (Stay)
            return 0 if current_pressure >= max_other_pressure else 1         

    def get_action(self, state, epsilon, curr_phase):
        if self.mode == 'train':
            self.retrieve_weights('online')

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.networks['actor_app'](state_tensor).squeeze(0)
        
        action_dist = action_probs.cpu().numpy()
        
        # --- GUIDED EXPLORATION ---
        action_rule = self.rule_based_attack_guide(state, 'minPressure', curr_phase) 

        # if (self.mode == 'train') and (np.random.rand() < epsilon):
        #     action = action_rule
        #     is_policy_action = False 
        # else:
        action = np.random.choice(np.arange(len(action_dist)), p=action_dist)
        is_policy_action = True 
        
        old_prob_1 = action_dist[action]

        if action == action_rule:
            self.same_action_cnt += 1
        self.total_action_cnt += 1
        
        print("##########RL agent action:", action, "Rule action:", action_rule, "Follow rate:",round(self.same_action_cnt/self.total_action_cnt,2) )

        return [action, action_dist, old_prob_1, action_rule, is_policy_action]
    
    def train_batch(self, sample_batch): 
        print("#####sample_batch length",len(sample_batch))
        
        # Unpack including old_values for Critic Clipping
        (states_np, actions_1, old_probs_1, s_effs, 
         returns_delay, returns_jsma, 
         norm_advantages_delay, norm_advantages_jsma, 
         is_policy_actions,
         old_values_delay_np, old_values_jsma_np) = self.process_batch(sample_batch)

        current_updates = self.rl_stats['updates']

        # =========================================================
        # PHASED TRAINING CONFIGURATION
        # =========================================================
        PRETRAIN_PHASE_LEN = 20 
        
        if current_updates < PRETRAIN_PHASE_LEN:
            # PHASE 1: PRE-TRAINING (BC Only)
            bc_weight = 0.0 #1.0
            ppo_weight = 1.0  
            self.eta = 0.3   
            new_beta = 0.1 
            
            if current_updates % 10 == 0:
                print(f"Update {current_updates}: PHASE 1 - PRE-TRAINING (Cloning Teacher...)")
        else:
            # PHASE 2: RL FINE-TUNING
            bc_weight = 0.0 #0.1   
            ppo_weight = 1.0  
            self.eta = 0.1   
            new_beta = 0.05
            
            if current_updates == PRETRAIN_PHASE_LEN:
                 print("!!! PHASE 2: SWITCHING TO RL FINE-TUNING !!!")
        # =========================================================

        new_lr = 5e-5
        self.delay_lambda = 1.0
        self.jsma_lambda = 0.1

        if self._last_applied_beta is None or new_beta != self._last_applied_beta:
            self.networks['actor_app'].set_beta(new_beta)
            self._last_applied_beta = new_beta
            
        if not hasattr(self, '_last_applied_lr'): self._last_applied_lr = None
        if new_lr != self._last_applied_lr:
            print(f"--- Update {current_updates}: LR={new_lr:.1e}, Beta={new_beta:.3f}, BC_Weight={bc_weight:.2f}, PPO_Weight={ppo_weight:.2f} ---")
            for param_group in self.networks['actor_app'].optimizer.param_groups: param_group['lr'] = new_lr
            for param_group in self.networks['critic'].optimizer.param_groups: param_group['lr'] = new_lr
            self._last_applied_lr = new_lr

        # --- SINGLE ADVANTAGE ---
        # Since JSMA is guaranteed, we rely solely on delay advantage
        advantages_final = (self.delay_lambda * norm_advantages_delay) + (self.jsma_lambda * norm_advantages_jsma)

        # Convert to Tensors
        states_t = torch.tensor(states_np, dtype=torch.float32)
        actions_1_t = torch.tensor(actions_1, dtype=torch.long)
        old_probs_1_t = torch.tensor(old_probs_1, dtype=torch.float32)
        
        s_effs_t = torch.tensor(s_effs, dtype=torch.float32)
        
        is_policy_t = torch.tensor(is_policy_actions, dtype=torch.bool) 
        advantages_final_t = torch.tensor(advantages_final, dtype=torch.float32) 
        returns_delay_t = torch.tensor(returns_delay, dtype=torch.float32).unsqueeze(1) 
        returns_jsma_t = torch.tensor(returns_jsma, dtype=torch.float32).unsqueeze(1)

        old_values_delay_t = torch.tensor(old_values_delay_np, dtype=torch.float32).unsqueeze(1)
        old_values_jsma_t = torch.tensor(old_values_jsma_np, dtype=torch.float32).unsqueeze(1)
        
        N = states_t.size(0)
        
        # =================================================================
        # CRITIC TRAINING (Single Head)
        # =================================================================
        CRITIC_EPOCHS = 5  
        CLIP_RANGE_VALUE = 0.2
        
        total_critic_loss = 0
        num_minibatches = 0 

        for c_epoch in range(CRITIC_EPOCHS): 
            indices = torch.randperm(N)
            
            for start in range(0, N, self.MINIBATCH_SIZE):
                end = start + self.MINIBATCH_SIZE
                minibatch_indices = indices[start:end]
                
                mb_states = states_t[minibatch_indices]
                mb_returns_delay = returns_delay_t[minibatch_indices]
                mb_returns_jsma = returns_jsma_t[minibatch_indices]
                mb_old_values_delay = old_values_delay_t[minibatch_indices]
                mb_old_values_jsma = old_values_jsma_t[minibatch_indices]

                self.networks['critic'].optimizer.zero_grad()
                
                # Single value prediction
                values_delay_pred, values_jsma_pred = self.networks['critic'](mb_states)
                
                # --- Delay Loss with Clipping ---
                v_pred_delay_clipped = mb_old_values_delay + torch.clamp(
                    values_delay_pred - mb_old_values_delay, 
                    -CLIP_RANGE_VALUE, 
                    CLIP_RANGE_VALUE
                )
                loss_delay_1 = (values_delay_pred - mb_returns_delay).pow(2)
                loss_delay_2 = (v_pred_delay_clipped - mb_returns_delay).pow(2)
                loss_delay = torch.max(loss_delay_1, loss_delay_2).mean()

                # --- JSMA Loss with Clipping ---
                v_pred_jsma_clipped = mb_old_values_jsma + torch.clamp(
                    values_jsma_pred - mb_old_values_jsma, 
                    -CLIP_RANGE_VALUE, 
                    CLIP_RANGE_VALUE
                )
                loss_jsma_1 = (values_jsma_pred - mb_returns_jsma).pow(2)
                loss_jsma_2 = (v_pred_jsma_clipped - mb_returns_jsma).pow(2)
                loss_jsma = torch.max(loss_jsma_1, loss_jsma_2).mean()
                
                # Combine losses (simple average or weighted)
                critic_loss = 0.5 * (loss_delay + loss_jsma)
                
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.networks['critic'].parameters(), 0.5) 
                self.networks['critic'].optimizer.step()

                if c_epoch == CRITIC_EPOCHS - 1: 
                    total_critic_loss += critic_loss.item()
                    num_minibatches += 1
        
        log_dir_path = get_fp(self.main_args, 'log')
        check_and_make_dir(log_dir_path)
        self.fp_loss_history = os.path.join(log_dir_path, str(self.tsc_id) + '_att_critic_loss_history.csv')
        avg_critic_loss = total_critic_loss / num_minibatches if num_minibatches > 0 else 0
        write_line_to_csv(self.fp_loss_history, [avg_critic_loss])  
        
        # =================================================================
        # ACTOR TRAINING
        # =================================================================
        ACTOR_EPOCHS = 10
        total_actor_loss = 0
        num_minibatches = 0 

        for epoch in range(ACTOR_EPOCHS): 
            indices = torch.randperm(N)
            
            for start in range(0, N, self.MINIBATCH_SIZE):
                end = start + self.MINIBATCH_SIZE
                minibatch_indices = indices[start:end]
                
                mb_states = states_t[minibatch_indices]
                mb_actions_1 = actions_1_t[minibatch_indices]
                mb_old_probs_1 = old_probs_1_t[minibatch_indices]
                mb_s_effs = s_effs_t[minibatch_indices] # Passed as 1.0s
                mb_advantages_final = advantages_final_t[minibatch_indices]
                mb_is_policy = is_policy_t[minibatch_indices]

                self.networks['actor_app'].optimizer.zero_grad()
                
                policy_indices = torch.nonzero(mb_is_policy, as_tuple=True)[0]
                rule_indices = torch.nonzero(~mb_is_policy, as_tuple=True)[0]
                
                loss_sum = 0.0
                
                # 1. PPO Loss
                if (len(policy_indices) > 0) and (ppo_weight > 0):
                    ppo_loss = self.networks['actor_app'].calculate_loss(
                        mb_states[policy_indices], 
                        mb_actions_1[policy_indices], 
                        mb_old_probs_1[policy_indices], 
                        mb_advantages_final[policy_indices], 
                        mb_s_effs[policy_indices], # Effectively ignored/always 1
                        self.eta,             
                        epsilon=self.epsilon_PPO
                    )
                    loss_sum += (ppo_weight * ppo_loss)

                # 2. BC Loss
                if (len(rule_indices) > 0) and (bc_weight > 0):
                    action_probs = self.networks['actor_app'](mb_states[rule_indices])
                    bc_loss = F.nll_loss(
                        torch.log(action_probs + 1e-8), 
                        mb_actions_1[rule_indices]
                    )
                    loss_sum += (bc_weight * bc_loss)

                if loss_sum != 0.0:
                    loss_sum.backward()
                    torch.nn.utils.clip_grad_norm_(self.networks['actor_app'].parameters(), 0.5)
                    self.networks['actor_app'].optimizer.step()
                    
                    if epoch == ACTOR_EPOCHS - 1:
                        total_actor_loss += loss_sum.item()
                        num_minibatches += 1

        self.fp_loss_history = os.path.join(log_dir_path, str(self.tsc_id) + '_actor_1' + '_loss_history.csv')
        avg_actor_loss = total_actor_loss / num_minibatches if num_minibatches > 0 else 0
        write_line_to_csv(self.fp_loss_history, [avg_actor_loss])  
        
        self.rl_stats['updates'] += 1
        if self.rl_stats['updates'] % 100 == 0:
            print("Current updates:", self.rl_stats['updates'])
            
        self.send_weights('online')

    def process_batch(self, sample_batch):     
        states, next_states, actions_1, old_probs_1, rewards_delay, rewards_jsma, s_effs = [], [], [], [], [], [], []
        is_policy_actions = [] 
        
        for trajectory in sample_batch:
            for exp in trajectory:
                rewards_delay.append(exp['r_delay'])
                rewards_jsma.append(exp['r_jsma'])
                states.append(exp['s'])
                actions_1.append(exp['a'][0])
                old_probs_1.append(exp['a'][2])
                is_policy_actions.append(exp['a'][4]) 
                next_states.append(exp['next_s'])
                s_effs.append(exp.get('s_eff', 1.0)) 

        states_np = np.stack(states)
        next_states_np = np.stack(next_states)
        rewards_delay_np = np.array(rewards_delay, dtype=np.float32)
        rewards_jsma_np = np.array(rewards_jsma, dtype=np.float32)
        is_policy_actions_np = np.array(is_policy_actions, dtype=bool)

        mean_reward_delay = np.mean(rewards_delay_np)
        mean_reward_jsma = np.mean(rewards_jsma_np)

        if not hasattr(self, 'reward_history'):
            self.reward_history = []
        self.reward_history.append(mean_reward_delay) 

        log_dir_path = get_fp(self.main_args, 'log')
        check_and_make_dir(log_dir_path)

        self.fp_reward_history_delay = os.path.join(log_dir_path, str(self.tsc_id) + '_actor_delay_reward_history.csv')
        write_line_to_csv(self.fp_reward_history_delay, [mean_reward_delay])

        self.fp_reward_history_jsma = os.path.join(log_dir_path, str(self.tsc_id) + '_actor_jsma_reward_history.csv')
        write_line_to_csv(self.fp_reward_history_jsma, [mean_reward_jsma])

        
        states_t = torch.tensor(states_np, dtype=torch.float32)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32)
        
        with torch.no_grad():
            state_values_delay, state_values_jsma = self.networks['critic'](states_t)
            next_state_values_delay, next_state_values_jsma = self.networks['critic'](next_states_t)
            
            state_values_delay = state_values_delay.squeeze(1).cpu().numpy()
            state_values_jsma = state_values_jsma.squeeze(1).cpu().numpy()
            next_state_values_delay = next_state_values_delay.squeeze(1).cpu().numpy()
            next_state_values_jsma = next_state_values_jsma.squeeze(1).cpu().numpy()
    
        # --- Calculate GAE for Delay ---
        advantages_delay = np.zeros_like(rewards_delay_np, dtype=np.float32)
        start_index = 0
        for trajectory in sample_batch:
            traj_len = len(trajectory)
            end_index = start_index + traj_len
            last_gae_lambda = 0
            for t in reversed(range(traj_len)):
                T = start_index + t
                delta = rewards_delay_np[T] + self.gamma * next_state_values_delay[T] - state_values_delay[T]
                advantages_delay[T] = delta + self.gamma * self.gae_lambda * last_gae_lambda
                last_gae_lambda = advantages_delay[T]
            start_index = end_index
        
        # --- Calculate GAE for JSMA ---
        advantages_jsma = np.zeros_like(rewards_jsma_np)
        start_index = 0
        for trajectory in sample_batch:
            traj_len = len(trajectory)
            end_index = start_index + traj_len
            last_gae_lambda = 0
            for t in reversed(range(traj_len)):
                T = start_index + t
                delta = rewards_jsma_np[T] + self.gamma * next_state_values_jsma[T] - state_values_jsma[T]
                advantages_jsma[T] = delta + self.gamma * self.gae_lambda * last_gae_lambda
                last_gae_lambda = advantages_jsma[T]
            start_index = end_index

        returns_delay = advantages_delay + state_values_delay
        returns_jsma = advantages_jsma + state_values_jsma

        norm_advantages_delay = (advantages_delay - np.mean(advantages_delay)) / (np.std(advantages_delay) + 1e-8)
        norm_advantages_jsma = (advantages_jsma - np.mean(advantages_jsma)) / (np.std(advantages_jsma) + 1e-8)
        
        # NOTE: returning state_values_delay and state_values_jsma as "old_values" for clipping
        return (states_np, actions_1, old_probs_1, s_effs, 
                returns_delay, returns_jsma, 
                norm_advantages_delay, norm_advantages_jsma,
                is_policy_actions_np,
                state_values_delay, state_values_jsma)        
    
    def send_weights(self, nettype):
        self.rl_stats[nettype]['actor_app'] = self.networks['actor_app'].state_dict()
        self.rl_stats[nettype]['critic'] = self.networks['critic'].state_dict()

    def retrieve_weights(self, nettype):
        if (self.rl_stats[nettype]['actor_app'] is not None): 
            self.networks['actor_app'].load_state_dict(self.rl_stats[nettype]['actor_app'])