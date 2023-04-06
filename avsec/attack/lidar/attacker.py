# -*- coding: utf-8 -*-
# @Author: spencer@primus
# @Date:   2022-05-16
# @Last Modified by:   spencer@primus
# @Last Modified time: 2022-09-16

import argparse
import time
from seca.attack.lidar import scheduler, monitor, executor
from seca.attack.types import Attacker


# ================================================================
# NEW ATTACK MODELS
# ================================================================

def get_sensor_details(dataset):
    """This is assumed known"""
    if dataset == 'kitti':
        sensor_name = 'hdl-64e'
        sensor_rate = 10
    elif dataset == 'nuscenes':
        sensor_name = 'hdl-32e'
        sensor_rate = 20
    else:
        raise NotImplementedError(dataset)
    return sensor_name, sensor_rate


class FalsePositiveObjectAttacker(Attacker):
    def __init__(self, awareness='none', framerate=10, dataset='kitti'):
        assert awareness == 'none'
        monitor_ = monitor.NaiveSceneMonitor(dataset)
        if dataset == 'kitti':
            dt_stable = 2
            attack_profile = 'jerk'
            init_range = 22
            final_range = 4
        elif dataset == 'nuscenes':
            dt_stable = 3
            attack_profile = 'linear'
            init_range = 15
            final_range = 5
        else:
            raise NotImplementedError(dataset)
        scheduler_ = scheduler.NaiveObjectStopScheduler(framerate=framerate,
            dt_stable=dt_stable, attack_profile=attack_profile,
            init_range=init_range, final_range=final_range)
        executor_ = executor.PointsAsObjectExecutor(*get_sensor_details(dataset))
        super().__init__(monitor_, scheduler_, executor_)


class ReplayAttacker(Attacker):
    def __init__(self, awareness='none', framerate=10, dataset='kitti', reverse=False):
        assert awareness == 'none'
        if dataset == 'kitti':
            dt_stable = 4
            dt_attack = 30  # any large number
            dt_repeat = 0.5
        elif dataset == 'nuscenes':
            dt_stable = 8
            dt_attack = 30  # any large number
            dt_repeat = 1
        else:
            raise NotImplementedError(dataset)
        monitor_ = monitor.PassthroughMonitor()
        scheduler_ = scheduler.ReplayScheduler(dt_burnin=0, dt_stable=dt_stable,
            dt_attack=dt_attack, framerate=framerate, dt_repeat=dt_repeat)
        if reverse:
            executor_ = executor.ReverseReplayExecutor(*get_sensor_details(dataset))
        else:
            executor_ = executor.ReplayExecutor(*get_sensor_details(dataset))
        super().__init__(monitor_, scheduler_, executor_)


class ReverseReplayAttacker(ReplayAttacker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, reverse=True)


class RemoveObjectAttacker(Attacker):
    def __init__(self, dataset, framerate, awareness='high',
            gpu_ID=0, save_folder='', save=False):
        monitor_ = monitor.FullSceneMonitor(dataset, framerate, awareness, gpu_ID, save_folder, save)
        if dataset == 'kitti':
            dt_stable = 1.5
            dt_attack = 4
        elif dataset == 'nuscenes':
            dt_stable = 2.5
            dt_attack = 4.5
        else:
            raise NotImplementedError(dataset)
        scheduler_ = scheduler.FrustumObjectStopScheduler(framerate=framerate, dt_stable=dt_stable, dt_attack=dt_attack)
        executor_ = executor.PointsAsBackgroundExecutor(*get_sensor_details(dataset))
        super().__init__(monitor_, scheduler_, executor_)


class FrustumTranslateAttacker(Attacker):
    def __init__(self, dataset, framerate, awareness='high',
            gpu_ID=0, save_folder='', save=False):
        monitor_ = monitor.FullSceneMonitor(dataset, framerate, awareness, gpu_ID, save_folder, save)
        if dataset == 'kitti':
            dt_stable = 1.5
            dt_attack = 4
        elif dataset == 'nuscenes':
            dt_stable = 2.5
            dt_attack = 4.5
        else:
            raise NotImplementedError(dataset)
        scheduler_ = scheduler.FrustumObjectStopScheduler(framerate=framerate, dt_stable=dt_stable, dt_attack=dt_attack)
        executor_ = executor.FrustumTranslateExecutor(*get_sensor_details(dataset))
        super().__init__(monitor_, scheduler_, executor_)


class PollingAttacker():
    def __init__(self, attacker_name, HOST, PORT) -> None:
        pass

    def poll(self):
        while True:
            time.sleep(0.05)
            pass


def main(args):
    # -- start up attacker


    # -- wait in a polling loop
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the attacker configuration')

    args = parser.parse_args()
    main(args)