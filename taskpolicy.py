"""
Created by: Matthew Zhang
Date: 2018-03-24, 4:28PM

Learning Through Policy Sketches
"""
import tensorflow as tf

class Taskpolicy():
    def __init__(self, subpolicy, critic, icm, params):
        self.name = None
        self.params = params
        self.subpolicy = subpolicy
        self.critic = critic
        self.icm = icm
        self.actor_iterator = 0
        self.general_iterator = 0
        
    def _rollout(self, session, writer, task, log_vis, curiosity, visitation):
        transitions = []
        total_reward = 0
        for sketch in task.sketch:
            subpol_rew, visitation = self.subpolicy.semi_rollout(session, sketch,
                                                     task.environment, transitions,
                                                     log_vis, visitation)
            total_reward += subpol_rew
        self.icm.run(session, transitions, curiosity)   
        
        if total_reward >= self.params["write_thresh"]:
            summary = tf.Summary()
            summary.value.add(tag='Total Reward', simple_value = total_reward)
            writer.add_summary(summary, self.general_iterator)
        
        task.environment.reset()
        return total_reward, transitions, visitation
        
    def _train(self, session, writer, transitions, task, pol = False):
        if pol:
            for sketch in task.sketch:
                summary = self.subpolicy.train(session, transitions, sketch)
                writer.add_summary(summary, self.actor_iterator)
                self.actor_iterator += 1
        
        else:
            t_states, t_returns = zip(*[(transition[0], transition[2]) for transition in transitions])
            t_baselines, critic_summary = self.critic.evaluate(session, self.name, t_states, t_returns)
            writer.add_summary(critic_summary, self.general_iterator)
    
            for transition, baseline in zip(transitions, t_baselines):
                transition.insert(3, baseline[0])
            transitions = [tuple(transition) for transition in transitions]
            
            icm_summary = self.icm.train(session, transitions)
            writer.add_summary(icm_summary, self.general_iterator)
            self.general_iterator += 1
            return transitions
        
    def run(self, session, writer, task, train = True, log_vis = False, 
            curiosity = True, visitation = None):
        self.name = task.name
        reward = 0
        
        transitions = []
        for _ in range(self.params["rollout"]):
            ro_reward, ro_transitions, visitation = self._rollout(session, writer, task,
                                                        log_vis, curiosity,
                                                        visitation)
            reward += ro_reward
            ro_transitions = self._train(session, writer, ro_transitions, task)
            transitions += ro_transitions
            
        if train:
            transitions = self._train(session, writer, transitions, task, pol = True)
        
        return reward, visitation