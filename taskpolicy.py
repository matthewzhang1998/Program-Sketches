"""
Created by: Matthew Zhang
Date: 2018-03-24, 4:28PM

Learning Through Policy Sketches
"""
class Taskpolicy():
    def __init__(self, subpolicy, critic, params):
        self.name = None
        self.params = params
        self.subpolicy = subpolicy
        self.critic = critic
        self.actor_iterator = 0
        self.critic_iterator = 0
        
    def _rollout(self, session, task):
        transitions = []
        total_reward = 0
        for sketch in task.sketch:
            subpol_rew = self.subpolicy.semi_rollout(session, sketch, task.environment, transitions)
            total_reward += subpol_rew
        task.environment.reset()
        return total_reward, transitions
        
    def _train(self, session, writer, transitions, task):
        t_states, t_returns = zip(*[(transition[0], transition[2]) for transition in transitions])
        t_baselines, summary = self.critic.evaluate(session, self.name, t_states, t_returns)
        writer.add_summary(summary, self.critic_iterator)
        self.critic_iterator += 1

        for transition, baseline in zip(transitions, t_baselines):
            transition.insert(3, baseline[0])
        transitions = [tuple(transition) for transition in transitions]
        for sketch in task.sketch:
            summary = self.subpolicy.train(session, transitions, sketch)
            writer.add_summary(summary, self.actor_iterator)
            self.actor_iterator += 1
        return transitions
        
    def run(self, session, writer, task):
        self.name = task.name
        
        reward, transitions = self._rollout(session, task)
        transitions = self._train(session, writer, transitions, task)
        
        return reward