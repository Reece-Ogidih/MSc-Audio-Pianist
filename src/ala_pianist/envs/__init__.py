"""Environment wrappers for ALA Pianist."""

from ala_pianist.envs.one_hand_env import OneHandRoboPianistEnv

ALAOneHandEnv = OneHandRoboPianistEnv

__all__ = ["ALAOneHandEnv", "OneHandRoboPianistEnv"]
