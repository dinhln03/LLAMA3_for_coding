from myelin.utils import CallbackList, Experience


class RLInteraction:
    """An episodic interaction between an agent and an environment."""

    def __init__(self, env, agent, callbacks=None, termination_conditions=None):
        self.env = env
        self.agent = agent
        self.callbacks = CallbackList(callbacks)
        if termination_conditions is None:
            self.termination_conditions = [lambda a: False]
        else:
            self.termination_conditions = termination_conditions
        self.episode = 0
        self.step = 0

    @property
    def info(self):
        return {
            'episode': self.episode,
            'step': self.step
        }

    def should_continue(self):
        for termination_condition in self.termination_conditions:
            if termination_condition(self.info):
                print(termination_condition)
                return False
        return True

    def start(self):
        """Starts agent-environment interaction."""
        self.callbacks.on_interaction_begin()
        while self.should_continue():
            self.callbacks.on_episode_begin(self.episode)
            self.env.reset()
            self.step = 0
            while not self.env.is_terminal():
                self.callbacks.on_step_begin(self.step)
                state = self.env.get_state()
                action = self.agent.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                experience = Experience(state, action, reward, next_state, done)
                self.agent.update(experience)
                self.callbacks.on_step_end(self.step)
                self.step += 1
            self.callbacks.on_episode_end(self.episode, self.step)
            self.episode += 1
        self.callbacks.on_interaction_end(self.episode)
