from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types

from ryu.lib.packet import in_proto
from ryu.lib.packet import ipv4
from ryu.lib.packet import icmp
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.lib.packet import arp


from ryu.lib.packet import arp
from ryu.lib import hub
import csv
import time
import statistics
from collections import deque

from svm import SVM


APP_TYPE = 0
#0 datacollection, 1 ddos detection

PREVENTION = 1
# ddos prevention

#TEST_TYPE is applicable only for data collection
#0  normal traffic, 1 attack traffic
TEST_TYPE = 0

#data collection time interval in seconds
INTERVAL = 3

gflows = []
old_ssip_len = 0
prev_flow_count = 0
FLOW_SERIAL_NO = 0
iteration = 0

# Queuing parameters
queue = deque()  # Queue for storing packets during mitigation
MAX_QUEUE_SIZE = 100  # Maximum queue size
DROPPED_PACKETS_FILE = 'dropped_packets.csv'


def get_flow_number():
    global FLOW_SERIAL_NO
    FLOW_SERIAL_NO += 1
    return FLOW_SERIAL_NO


def init_portcsv(dpid):
    fname = f"switch_{dpid}_data.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    header = ["time", "sfe", "ssip", "rfip", "type"]
    writ.writerow(header)


def init_flowcountcsv(dpid):
    fname = f"switch_{dpid}_flowcount.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    header = ["time", "flowcount"]
    writ.writerow(header)


def update_flowcountcsv(dpid, row):
    fname = f"switch_{dpid}_flowcount.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    writ.writerow(row)


def update_portcsv(dpid, row):
    fname = f"switch_{dpid}_data.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    row.append(str(TEST_TYPE))
    writ.writerow(row)


def update_resultcsv(row):
    fname = "result.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    row.append(str(TEST_TYPE))
    writ.writerow(row)


def log_dropped_packet(packet_info):
    with open(DROPPED_PACKETS_FILE, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(packet_info)


class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.flow_thread = hub.spawn(self._flow_monitor)
        self.datapaths = {}
        self.mitigation = 0
        self.svmobj = None
        self.arp_ip_to_port = {}
        if APP_TYPE == 1:
            self.svmobj = SVM()

    def _flow_monitor(self):
        hub.sleep(5)
        while True:
            for dp in self.datapaths.values():
                self.request_flow_metrics(dp)
            hub.sleep(INTERVAL)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        flow_serial_no = get_flow_number()
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                           ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions, flow_serial_no)

        init_portcsv(datapath.id)
        init_flowcountcsv(datapath.id)

    def request_flow_metrics(self, datapath):
        ofp = datapath.ofproto
        ofp_parser = datapath.ofproto_parser
        req = ofp_parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    def _speed_of_flow_entries(self, flows):
        global prev_flow_count
        curr_flow_count = len(flows)
        sfe = curr_flow_count - prev_flow_count
        prev_flow_count = curr_flow_count
        return sfe

    def _speed_of_source_ip(self, flows):
        global old_ssip_len
        ssip = set()
        for flow in flows:
            for key, val in flow.match.items():
                if key == "ipv4_src":
                    ssip.add(val)
        cur_ssip_len = len(ssip)
        ssip_result = cur_ssip_len - old_ssip_len
        old_ssip_len = cur_ssip_len
        return ssip_result

    def _ratio_of_flowpair(self, flows):
        flow_count = len(flows) - 1  # Excluding the table miss entry
        collaborative_flows = {}
        for flow in flows:
            srcip = dstip = None
            for key, val in flow.match.items():
                if key == "ipv4_src":
                    srcip = val
                if key == "ipv4_dst":
                    dstip = val
            if srcip and dstip:
                fwdflowhash = f"{srcip}_{dstip}"
                revflowhash = f"{dstip}_{srcip}"
                if fwdflowhash not in collaborative_flows:
                    collaborative_flows[fwdflowhash] = {}
                else:
                    collaborative_flows[revflowhash][fwdflowhash] = 1
        if flow_count != 0:
            rfip = float(len(collaborative_flows)) / flow_count
            return rfip
        return 1.0

    @set_ev_cls([ofp_event.EventOFPFlowStatsReply], MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        global gflows, iteration
        t_flows = ev.msg.body
        flags = ev.msg.flags
        dpid = ev.msg.datapath.id
        gflows.extend(t_flows)
        if flags == 0:
            sfe = self._speed_of_flow_entries(gflows)
            ssip = self._speed_of_source_ip(gflows)
            rfip = self._ratio_of_flowpair(gflows)

            if APP_TYPE == 1:
                result = self.svmobj.classify([sfe, ssip, rfip])
                if '1' in result:  # Attack detected
                    print("Attack Traffic detected")
                    self.mitigation = 1
                    if PREVENTION == 1:
                        print("Mitigation Started")
                        # Start queuing packets
                        self.start_queuing(dpid)
                if '0' in result:  # Normal traffic
                    print("It's Normal Traffic")

            else:
                t = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
                row = [t, str(sfe), str(ssip), str(rfip)]
                self.logger.info(row)
                update_portcsv(dpid, row)
                update_resultcsv([str(sfe), str(ssip), str(rfip)])
            gflows = []
            t = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
            update_flowcountcsv(dpid, [t, str(prev_flow_count)])

    def add_flow(self, datapath, priority, match, actions, serial_no, buffer_id=None, idletime=0, hardtime=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, cookie=serial_no, buffer_id=buffer_id,
                                    idle_timeout=idletime, hard_timeout=hardtime,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, cookie=serial_no, priority=priority,
                                    idle_timeout=idletime, hard_timeout=hardtime,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    def block_port(self, datapath, portnumber):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(in_port=portnumber)
        actions = []
        flow_serial_no = get_flow_number()
        self.add_flow(datapath, 100, match, actions, flow_serial_no, hardtime=120)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        dst = eth.dst
        src = eth.src
        self.mac_to_port.setdefault(datapath.id, {})

        if src in self.mac_to_port[datapath.id]:
            self.mac_to_port[datapath.id][src] += 1
        else:
            self.mac_to_port[datapath.id][src] = 1

        if dst in self.mac_to_port[datapath.id]:
            self.mac_to_port[datapath.id][dst] += 1
        else:
            self.mac_to_port[datapath.id][dst] = 1

        if self.mitigation == 1:
            # Queue packets for later processing during DDoS mitigation
            self.queue_packet(msg)
            return

        # Learn a MAC address to avoid FLOOD next time.
        self.mac_to_port[datapath.id][src] = in_port

        # Install a flow to avoid packet_in next time
        self._install_flow(datapath, in_port, eth, src)

    def _install_flow(self, datapath, in_port, eth, src):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
        match = parser.OFPMatch(in_port=in_port, eth_dst=eth.dst)
        flow_serial_no = get_flow_number()
        self.add_flow(datapath, 1, match, actions, flow_serial_no)

    def start_queuing(self, dpid):
        self.logger.info("Starting packet queuing")
        while self.mitigation == 1:
            if len(queue) > 0:
                packet_info = queue.popleft()  # Process the next packet
                self.process_queued_packet(packet_info)
            hub.sleep(0.1)

    def queue_packet(self, packet_info):
        if len(queue) < MAX_QUEUE_SIZE:
            queue.append(packet_info)
            self.logger.info(f"Packet queued: {packet_info}")
        else:
            # Drop the packet if the queue is full
            log_dropped_packet([time.time(), packet_info])

    def process_queued_packet(self, packet_info):
        # Here, you can add your logic to handle the queued packets
        # For now, just simulate processing the packet
        self.logger.info(f"Processing queued packet: {packet_info}")
