import os, sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

class TrafficMetrics:
    def __init__(self, _id, incoming_lanes, netdata, metric_args, mode):
        self.metrics = {}
        if 'delay' in metric_args:
            lane_lengths = {lane:netdata['lane'][lane]['length'] for lane in incoming_lanes}
            lane_speeds = {lane:netdata['lane'][lane]['speed'] for lane in incoming_lanes}
            self.metrics['delay'] = DelayMetric(_id, incoming_lanes, mode, lane_lengths, lane_speeds )

        if 'main_delay' in metric_args:
            lane_lengths = {lane:netdata['lane'][lane]['length'] for lane in incoming_lanes}
            lane_speeds = {lane:netdata['lane'][lane]['speed'] for lane in incoming_lanes}
            self.metrics['main_delay'] = MainDelayMetric(_id, incoming_lanes, mode, lane_lengths, lane_speeds )
            if 'side_delay' in metric_args:
                self.metrics['side_delay'] = SideDelayMetric(_id, incoming_lanes, mode, lane_lengths, lane_speeds)

        if 'queue' in metric_args:
            self.metrics['queue'] = QueueMetric(_id, incoming_lanes, mode)

        # wz: add our metric
        if 'vehicle' in metric_args:
            self.metrics['vehicle'] = VehicleMetric(_id, incoming_lanes, mode, lane_lengths, lane_speeds)

        if 'intersec' in metric_args:
            self.metrics['intersec'] = IntersecMetric(_id, incoming_lanes, mode)


    def update(self, v_data, cv_data=None):
        for m in self.metrics:
            self.metrics[m].update(v_data, cv_data)

    def get_metric(self, metric):
        return self.metrics[metric].get_metric()

    def get_history(self, metric):
        return self.metrics[metric].get_history()

    def get_metric_lane(self, metric):
        if metric == 'delay':
            return self.metrics[metric].get_metric_lane()
        else:
            raise NotImplementedError("Stupid! We need Delay for each lane as state")

class TrafficMetric:
    def __init__(self, _id, incoming_lanes, mode):
        self.id = _id
        self.incoming_lanes = incoming_lanes
        self.history = []
        self.mode = mode

    def get_metric(self):
        pass

    def update(self):
        pass

    def get_history(self):
        return self.history

class DelayMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode, lane_lengths, lane_speeds):
        super().__init__( _id, incoming_lanes, mode)
        self.lane_travel_times = {lane:lane_lengths[lane]/float(lane_speeds[lane]) for lane in incoming_lanes}
        self.old_v = set()
        self.old_cv = set()
        self.v_info = {}
        self.t = 0
        self.delay = 0
        self.delay_cv = 0
        self.avg_delay = 0
        self.avg_delay_cv = 0
        self.incoming_lanes = incoming_lanes

    def get_v_delay(self, v):
        return (self.t - self.v_info[v]['t']) - self.lane_travel_times[self.v_info[v]['lane']]

    def get_metric(self):
        #calculate delay of vehicles on incoming lanes
        return self.delay, self.delay_cv, self.avg_delay, self.avg_delay_cv

    def get_metric_lane(self):
        #calculate delay of vehicles on incoming lanes
        return self.avg_delay_lane, self.avg_delay_cv_lane, self.avg_delay_uv_lane

    def update(self, v_data, cv_data=None):
        new_v = set()
        new_cv = set()

        #record start time and lane of new_vehicles
        for lane in self.incoming_lanes:
            for v in v_data[lane]:
                if v not in self.old_v:
                    self.v_info[v] = {}
                    self.v_info[v]['t'] = self.t
                    self.v_info[v]['lane'] = lane
            new_v.update(set(v_data[lane].keys())) # union: add new vehicle
            if cv_data is not None:
                new_cv.update(set(cv_data[lane].keys()))

        if self.mode == 'test':
            self.history.append(self.get_metric())

        #remove vehicles that have left incoming lanes
        remove_vehicles = self.old_v - new_v
        for v in remove_vehicles:
            del self.v_info[v]
        
        self.old_v = new_v
        self.old_cv = new_cv
        self.t += 1

        self.delay = 0
        self.delay_cv = 0
        for v in self.old_v:
            # calculate individual vehicle delay
            v_delay = self.get_v_delay(v)
            if v_delay > 0:
                self.delay += v_delay
                if v in self.old_cv:
                    self.delay_cv += v_delay

        self.avg_delay = self.delay / max(len(self.old_v), 1)
        self.avg_delay_cv = self.delay_cv / max(len(self.old_cv), 1)

        self.delay_cv_lane = {lane: 0 for lane in self.incoming_lanes}
        self.delay_uv_lane = {lane: 0 for lane in self.incoming_lanes}
        self.delay_lane = {lane: 0 for lane in self.incoming_lanes}
        self.lane_veh_count = {lane:0 for lane in self.incoming_lanes}
        self.lane_veh_cv_count = {lane:0 for lane in self.incoming_lanes}
        self.lane_veh_uv_count = {lane: 0 for lane in self.incoming_lanes}

        '''for v in self.old_cv:
            lane = self.v_info[v]['lane']
            v_delay = self.get_v_delay(v)
            if v_delay > 0:
                self.delay_cv_lane[lane] += v_delay
                self.lane_veh_cv_count[lane] += 1'''
        for v in self.old_v:
            if v in self.old_cv:
                lane = self.v_info[v]['lane']
                v_delay = self.get_v_delay(v)
                if v_delay > 0:
                    self.delay_cv_lane[lane] += v_delay
                    self.lane_veh_cv_count[lane] += 1
                    self.delay_lane[lane] += v_delay
                    self.lane_veh_count[lane] += 1
            else:
                lane = self.v_info[v]['lane']
                v_delay = self.get_v_delay(v)
                if v_delay > 0:
                    self.delay_uv_lane[lane] += v_delay
                    self.lane_veh_uv_count[lane] += 1
                    self.delay_lane[lane] += v_delay
                    self.lane_veh_count[lane] += 1

        self.avg_delay_uv_lane = {lane: (((self.delay_uv_lane[lane]/self.lane_veh_uv_count[lane])//self.lane_travel_times[lane]) if self.lane_veh_uv_count[lane] else 0 ) for lane in self.incoming_lanes }
        self.avg_delay_cv_lane = {lane: (((self.delay_cv_lane[lane]/self.lane_veh_cv_count[lane])//self.lane_travel_times[lane]) if self.lane_veh_cv_count[lane] else 0 ) for lane in self.incoming_lanes }
        self.avg_delay_lane = {lane: (((self.delay_lane[lane]/self.lane_veh_count[lane])//self.lane_travel_times[lane]) if self.lane_veh_count[lane] else 0 ) for lane in self.incoming_lanes }

        # print(self.t, self.delay, self.delay_cv)


class MainDelayMetric(DelayMetric):
    def __init__(self, _id, incoming_lanes, mode, lane_lengths, lane_speeds):
        main_incoming_lanes = set()
        for lane in incoming_lanes:
            if lane[:5] == 'Main.' or lane[:6] == '-Main.':
                main_incoming_lanes.add(lane)
        super().__init__(_id, main_incoming_lanes, mode, lane_lengths, lane_speeds)
        self.lane_travel_times = {lane: lane_lengths[lane] / float(lane_speeds[lane]) for lane in main_incoming_lanes}
        self.old_v = set()
        self.old_cv = set()
        self.v_info = {}
        self.t = 0
        self.delay = 0
        self.delay_cv = 0
        self.avg_delay = 0
        self.avg_delay_cv = 0
        self.incoming_lanes = main_incoming_lanes


class SideDelayMetric(DelayMetric):
    def __init__(self, _id, incoming_lanes, mode, lane_lengths, lane_speeds):
        side_incoming_lanes = set()
        for lane in incoming_lanes:
            if (lane[:5] != 'Main.') and (lane[:6] != '-Main.'):
                side_incoming_lanes.add(lane)
        super().__init__(_id, side_incoming_lanes, mode, lane_lengths, lane_speeds)
        self.lane_travel_times = {lane: lane_lengths[lane] / float(lane_speeds[lane]) for lane in side_incoming_lanes}
        self.old_v = set()
        self.old_cv = set()
        self.v_info = {}
        self.t = 0
        self.delay = 0
        self.delay_cv = 0
        self.avg_delay = 0
        self.avg_delay_cv = 0
        self.incoming_lanes = side_incoming_lanes


class QueueMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode):
        super().__init__( _id, incoming_lanes, mode)
        self.stop_speed = 0.1 # wz: originally, 0.3
        self.lane_queues = {lane:0 for lane in self.incoming_lanes}
        self.metric = 0

    def get_metric(self):
        return self.metric

    def update(self, v_data, cv_data=None):
        lane_queues = {}
        for lane in self.incoming_lanes:
            lane_queues[lane] = 0
            for v in v_data[lane]:
                if v_data[lane][v][traci.constants.VAR_SPEED] < self.stop_speed:
                    lane_queues[lane] += 1

        self.lane_queues = lane_queues
        if self.mode == 'test':
            self.history.append(self.get_metric())
        self.metric = sum([self.lane_queues[lane] for lane in self.lane_queues])
#### wz

class VehicleMetric(TrafficMetric):
    '''
    goal: to get that, for each Intersection, {vehicle ID: (waiting time, travel time, delay, EdgeID), ...}, ...}
    '''
    def __init__(self, _id, incoming_lanes, mode, lane_lengths, lane_speeds):
        super().__init__(_id, incoming_lanes, mode)
        self.lane_travel_times = {lane: lane_lengths[lane] / float(lane_speeds[lane]) for lane in incoming_lanes}
        self.old_v = set()
        self.v_info = {}
        self.stop_speed = 0.1 # wz
        self.t = 0

    def get_v_delay(self, v):
        return (self.t - self.v_info[v]['t']) - self.lane_travel_times[self.v_info[v]['lane']]

    def get_metric(self):
        # calculate delay of vehicles on incoming lanes
        return

    def update(self, v_data, cv_data=None):
        new_v = set()

        # record start time and lane of new_vehicles
        for lane in self.incoming_lanes:
            for v in v_data[lane]:
                if v not in self.old_v:
                    self.v_info[v] = {}
                    self.v_info[v]['id'] = v
                    self.v_info[v]['t'] = self.t
                    self.v_info[v]['lane'] = lane
                    self.v_info[v]['edge'] = v_data[lane][v][80] # 80 means road_id
                    self.v_info[v]['wait'] = 0 # initialize the waiting time; beyond the 150 m

                if v_data[lane][v][traci.constants.VAR_SPEED] < self.stop_speed:
                    self.v_info[v]['wait'] += 1

            new_v.update(set(v_data[lane].keys())) # add the moving vehicles

        # remove vehicles that have left incoming lanes
        leaving_vehicles = self.old_v - new_v

        # delay = 0
        for v in leaving_vehicles:
           self.left(v)
           if self.mode == 'test':
               self.history.append(self.v_info[v])

        self.old_v = new_v
        self.t += 1

    def left(self, v):
        self.v_info[v]['travel_time'] = self.t - self.v_info[v]['t']
        self.v_info[v]['delay'] = self.get_v_delay(v)


class IntersecMetric(TrafficMetric):
    '''
    Export the history:
    for each intersection ID, we have {lane Index: (queue length at time step 0, … at 1, … at 2...), ... }, ... }
    '''
    def __init__(self, _id, incoming_lanes, mode):
        super().__init__( _id, incoming_lanes, mode)
        self.stop_speed = 0.1 # from the repo
        self.lane_queues = {lane:[] for lane in self.incoming_lanes}
        # self.queues = {}

    def update(self, v_data, cv_data=None):
        lane_queues = {}
        for lane in self.incoming_lanes:
            lane_queues[lane] = 0
            for v in v_data[lane]:
                if v_data[lane][v][traci.constants.VAR_SPEED] < self.stop_speed:
                    lane_queues[lane] += 1

        # self.lane_queues = lane_queues
        for lane in self.lane_queues:
            self.lane_queues[lane].append(lane_queues[lane])

        if self.mode == 'test':
            self.history = self.lane_queues


class GlobalTrafficMetrics:
    def __init__(self, netdata, metric_args, mode):
        self.metrics = {}
        if 'flow' in metric_args:
            self.metrics['flow'] = FlowMetric(mode)
        if 'position' in metric_args:
            self.metrics['position'] = PositionMetric(mode)
        if 'route' in metric_args:
            self.metrics['route'] = RouteMetric(mode)

    def update(self, v_data, cv_data=None):
        for m in self.metrics:
            self.metrics[m].update(v_data, cv_data)

    def get_metric(self, metric):
        return self.metrics[metric].get_metric()

    def get_history(self, metric):
        return self.metrics[metric].get_history()


class GlobalTrafficMetric:
    def __init__(self, mode):
        self.history = []
        self.mode = mode

    def get_metric(self):
        pass

    def update(self):
        pass

    def get_history(self):
        return self.history


class FlowMetric(GlobalTrafficMetric):
    def __init__(self, mode):
        super().__init__(mode)
        self.v_metric = {}
        self.cv_metric = {}

    def get_metric(self):
        return self.v_metric, self.cv_metric

    def update(self, v_data, cv_data=None):
        self.v_metric = {}
        for lane in v_data:
            self.v_metric[lane] = len(v_data[lane])
        if cv_data is not None:
            self.cv_metric = {}
            for lane in cv_data:
                self.cv_metric[lane] = len(cv_data[lane])
        if self.mode == 'test':
            self.history.append(self.get_metric())


class PositionMetric(GlobalTrafficMetric):
    """
    Function: record positions for all vehicles to replay
    Formation: history[#timestamp](v, cv)[#vehicle_id] = (x, y)
    """
    def __init__(self, mode):
        super().__init__(mode)
        self.v_metric = {}
        self.cv_metric = {}

    def get_metric(self):
        return self.v_metric, self.cv_metric

    def update(self, v_data, cv_data=None):
        self.v_metric = {}
        for lane in v_data:
            for v in v_data[lane]:
                self.v_metric[v] = v_data[lane][v][traci.constants.VAR_POSITION]
        if cv_data is not None:
            self.cv_metric = {}
            for lane in cv_data:
                for cv in cv_data[lane]:
                    self.cv_metric[cv] = cv_data[lane][cv][traci.constants.VAR_POSITION]
        if self.mode == 'test':
            self.history.append(self.get_metric())


class RouteMetric(GlobalTrafficMetric):
    """
    Function: record positions for all vehicles to replay
    Formation: history[#timestamp](v, cv)[#vehicle_id] = (x, y)
    """
    def __init__(self, mode):
        super().__init__(mode)
        self.v_metric = {}
        self.cv_metric = {}

    def get_metric(self):
        return self.v_metric, self.cv_metric

    def update(self, v_data, cv_data=None):
        self.v_metric = {}
        for lane in v_data:
            for v in v_data[lane]:
                route = v_data[lane][v][traci.constants.VAR_EDGES]
                key = route[0] + '>' + route[-1]
                if key not in self.v_metric:
                    self.v_metric[key] = []
                self.v_metric[key].append((v, v_data[lane][v][traci.constants.VAR_POSITION]))
        if cv_data is not None:
            self.cv_metric = {}
            for lane in cv_data:
                for cv in cv_data[lane]:
                    route = cv_data[lane][cv][traci.constants.VAR_EDGES]
                    key = route[0] + '>' + route[-1]
                    if key not in self.cv_metric:
                        self.cv_metric[key] = []
                    self.cv_metric[key].append((cv, cv_data[lane][cv][traci.constants.VAR_POSITION]))
        if self.mode == 'test':
            self.history.append(self.get_metric())