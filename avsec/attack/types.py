
# ========================================================
# END-TO-END ATTACK LOGIC
# ========================================================

class Attacker():
    def __init__(self, monitor_, scheduler_, executor_):
        self.monitor = monitor_
        self.scheduler = scheduler_
        self.executor = executor_
        print(f'Initialized the following attacker:\n--Monitor:' +
        f'{self.monitor}\n--Scheduler: {self.scheduler}\n--Executor: {self.executor}')

    def __call__(self, data):
        diagnostics = {'monitor':None, 'scheduler':None, 'executor':None}
        ready_mon, info_mon, diagnostics['monitor'] = self.monitor(data)
        if ready_mon:
            ready_sch, info_sch, diagnostics['scheduler'] = self.scheduler(info_mon)
            if ready_sch:
                data, diagnostics['executor'] = self.executor(data, info_sch)
        else:
            info_sch = {}
        return data, {'monitor':info_mon, 'scheduler':info_sch}, diagnostics

# ========================================================
# SCENARIO MONITORING
# ========================================================

class Monitor():
    """An attacker's monitor that follows perception data over time"""
    def __init__(self):
        pass

    def __call__(self, data, **kwargs):
        """Ingest data and distill information"""
        self._ingest(data, **kwargs)
        return self._distill(**kwargs)

    def _ingest(self, data):
        """Intake some data to keep track of"""
        raise NotImplementedError

    def _distill(self):
        """Distill information"""
        raise NotImplementedError


# ========================================================
# ATTACK SCHEDULING
# ========================================================

class Scheduler():
    """An attacker's scheduler that determines when and where to do things"""
    def __init__(self, dt_burnin, dt_stable, dt_attack, framerate):
        self.ready = False
        # self.n_frames = 0
        self.n_frames_ready = 0
        self.dt_burnin = dt_burnin
        self.dt_stable = dt_stable
        self.dt_attack = dt_attack
        self.framerate = framerate
        self.dt = 1./framerate
        self.framerate = framerate
        self.n_frames_burnin = round(dt_burnin * framerate)
        self.n_frames_stable = round(dt_stable * framerate)
        self.n_frames_attack = round(dt_attack * framerate)
        assert self.n_frames_burnin >= 0
        assert self.n_frames_stable >= 0
        assert self.n_frames_attack >= 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'{type(self)} with parameters:\n' \
               f'----burnin: ({self.dt_burnin} s, {self.n_frames_burnin} frames); '\
               f'stable: ({self.dt_stable} s, {self.n_frames_stable} frames); '\
               f'attack: ({self.dt_attack} s, {self.n_frames_attack} frames)'

    def __call__(self, info):
        if self.n_frames_ready < self.n_frames_burnin:
            ready = False
            info_out = {}
            diagnostics = {}
            self.n_frames_ready += 1
        elif self.n_frames_ready <= (self.n_frames_burnin+self.n_frames_stable):
            info_out, diagnostics = self._schedule_stable(info)
            if info_out:
                ready = True
                self.n_frames_ready += 1
            else:
                ready = False
        elif self.n_frames_ready <= (self.n_frames_burnin+self.n_frames_stable+self.n_frames_attack):
            ready = True
            info_out, diagnostics = self._schedule_attack(info)
            self.n_frames_ready += 1
        else:
            # done with attack
            ready = False
            info_out = {}
            diagnostics = {}
            self.n_frames_ready += 1
        # print(info_out.get('pts_ground', None))
        return ready, info_out, diagnostics

    def _schedule_stable(self, info):
        raise NotImplementedError

    def _schedule_attack(self, info):
        raise NotImplementedError


# ========================================================
# ATTACK EXECUTION
# ========================================================

class Executor():
    """An attacker's execution module that implements the attack"""
    def __init__(self, sensor_name, sensor_rate):
        self.sensor_name = sensor_name
        self.sensor_rate = sensor_rate

    def __call__(self, data, info):
        """Process info, manipulate data"""
        return self._manipulate(data, info)

    def _manipulate(self, data, info):
        """Manipulate the data based on the info"""
        raise NotImplementedError
