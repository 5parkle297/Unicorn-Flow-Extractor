"""
@Created Time : 2022/11/22
@Author  : LiYao
@FileName: NfstreamPlugin.py
@Description:nfstream的插件
@Modified:
    :First modified
    :Modified content:040312整理上传
"""

from nfstream import NFPlugin
import binascii
import tiktoken
import struct


class TrafficExtractorPlugin(NFPlugin):
    """本插件实现获流数据包大小以及方向列表特征"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def decodeLoad(self, data):
        """
        二进制转换成字符串(pip install adafruit-circuitpython-binascii)
        :param data:二进制data
        :return: 字符串data
        """
        str = binascii.b2a_hex(data).decode()
        if str == '00':
            return None
        newLoad = ''
        i = 0
        for j in range(0, len(str), 2):
            newLoad += str[j:j + 2] + ' '
        newLoad = newLoad[:-1]
        return newLoad

    # pmy:

    def on_init(self, packet, flow):
        """初始化需要拿到的字段"""
        flow.udps.bi_pkt_size = str()
        flow.udps.bi_flow_pkt_direction = str()
        #flow.udps.bi_pkt_arrive_time = str()
        #flow.udps.bi_flow_syn = str()
        # pmy
        # flow.udps.bi_payload_tokens = []
        flow.udps.bi_payload_hex=[]
        flow.udps.packet_num=[]



    def on_update(self, packet, flow):

        flow.udps.bi_flow_pkt_direction += str(packet.direction) + ' '
        flow.udps.bi_pkt_size += str(packet.ip_size) + ' '

        # 只使用负载，为了和ET Bert对比
        ip_packet = packet.ip_packet  # 直接是 IP 层，Ethernet 已被 nfstream 移除
        protocol = packet.protocol  # 6 = TCP, 17 = UDP

        # 解析IP Header
        if len(ip_packet) < 20:
            return  # IP 包太短
        ip_header_len = (ip_packet[0] & 0x0F) * 4
        ip_header = ip_packet[:ip_header_len]
        ip_payload = ip_packet[ip_header_len:]

        # 解析 TCP/UDP Header
        if protocol == 6 and len(ip_payload) >= 20:  # TCP
            transport_header_len = (ip_payload[12] >> 4) * 4
        elif protocol == 17 and len(ip_payload) >= 8:  # UDP
            transport_header_len = 8
        else:
            return  # 非 TCP/UDP 或头不足

        transport_header = ip_payload[:transport_header_len]
        payload = ip_payload[transport_header_len:]

        # 丢弃无 payload 的包
        if len(payload) == 0:
            return  # 不处理无 payload 的包


        clean_payload = payload[:200]  # 保留前200字节

        # 转换为十六进制字符串（每个字节以两位十六进制表示）
        payload_hex = ''.join(f"{byte:02x}" for byte in clean_payload)

        # 将每个数据包的 hex 序列添加到 flow.udps.bi_payload_hex 列表中
        flow.udps.bi_payload_hex.append(payload_hex)  # 将当前数据包的 hex 序列追加到列表中

        if len(flow.udps.bi_payload_hex) > 100:
            flow.udps.bi_payload_hex = flow.udps.bi_payload_hex[:100]

        flow.udps.packet_num=len(flow.udps.bi_payload_hex)


    def on_expire(self, flow):
        """流过期时的标志"""
        if len(flow.udps.bi_payload_hex) == 0:
            return  # 丢弃该流
