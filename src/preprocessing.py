import numpy as np 
import collections
import cv2
import gym 

class RepeatActionMaxFrame(gym.Wrapper):
    def __init__(self, env, n_repeat_frames, clip_rewards=False, no_ops=0, fire_first=False):
        super(RepeatActionMaxFrame, self).__init__(env)
        self.shape = env.observation_space.shape
        self.frame_buffer = np.zeros((2, env.observation_space.shape), dtype=object)
        self.n_repeat_frames = n_repeat_frames
        self.clip_reward = clip_rewards
        self.no_ops = no_ops
        self.fire_first=  fire_first

    def step(self, action):
        total_reward = 0
        terminated = False
        
        for i in range(self.n_repeat_frames):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(reward, -1, 1)

            total_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if terminated:
                break
    
        max_frame = np.max(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, terminated, total_reward, truncated, info
    
    def reset(self):
        obs = self.env.reset()
        no_ops = n.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for _ in range (no_ops):
            _, _, terminated = self.env.step(0)
            if terminated:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, self.env.step(1)
            
        self.frame_buffer = np.zeros((2, self.env.observation_space.shape), dtype=object)
        self.frame_buffer[0] = obs

        return obs

class Preprocess(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(Preprocess, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        greyscale_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        resized_frame = cv2.resize(greyscale_frame, self.shape[1:], interpolation=cv2.INTER_AREA)

        new_obs = np.array(resized_frame, dtype = np.uint8).reshape(self.shape)
        scaled_new_obs = new_obs / 255.0

        return scaled_new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, stack_size):
        super(StackFrames, self).__init(env)
        self.stack_size = stack_size
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(stack_size, axis = 0),
                                                env.observation_space.high.repeat(stack_size, axis = 0), dtype=np.float32)
        self.stack = collections.deque(maxlen=stack_size)
    
    def reset(self):
        self.stack.clear()
        obs = self.env.reset()
        
        for i in range(self.stack_size):
            self.stack.append(obs)

        self.stack = np.array(self.stack).reshape(self.observation_space.low.shape)
        return self.stack
    
    def observation(self, obs):
        self.stack.append(obs)
        self.stack = np.array(self.stack).reshape(self.observation_space.low.shape)
        return self.stack   

def make_env(env_name, shape=(84, 84, 1), repeat=4, 
             clip_rewards=False, no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = Preprocess(env, shape)
    env = StackFrames(env, repeat)
    return env



        

    

