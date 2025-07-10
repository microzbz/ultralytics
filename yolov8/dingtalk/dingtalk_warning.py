import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
import psutil
from collections import deque
from datetime import datetime, timedelta

# ---------- 钉钉配置 ----------
ACCESS_TOKEN = '033266965c96fce6097fc29f835f4881ac83f99bd277a012e5ef17b5008d9cda'
SECRET = 'SEC8d7e36f0ea35681d4c1b6530aaacb7ee2b84cb83ef8e6372a0d6772805d1daa7'  # 可为空（不加签）
AT_MOBILES = ['19864791204']  # 要@的手机号码
IS_AT_ALL = False

# ---------- 告警阈值 ----------
CPU_THRESHOLD = 80      # CPU 使用率阈值
GPU_THRESHOLD = 10      # GPU 使用率阈值

# ---------- 滑动窗口配置 ----------
SAMPLE_INTERVAL = 5      # 每 N 秒采样一次
WINDOW_SECONDS = 60      # 检查最近 N 秒
MAX_SAMPLES = WINDOW_SECONDS // SAMPLE_INTERVAL
TRIGGER_COUNT = 3        # 超过阈值次数达到该值则告警

# ---------- 告警频率控制 ----------
ALERT_INTERVAL = 120     # 每隔 N 秒重复告警一次（持续超阈值）

# ---------- GPU 初始化 ----------
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
    nvmlInit()
    gpu_available = True
except:
    gpu_available = False
    print("[⚠] 无法初始化 GPU（pynvml），将跳过 GPU 监控。")

# ---------- 钉钉发送 ----------
def get_signed_url():
    if not SECRET:
        return f'https://oapi.dingtalk.com/robot/send?access_token={ACCESS_TOKEN}'
    timestamp = str(round(time.time() * 1000))
    secret_enc = SECRET.encode('utf-8')
    string_to_sign = f'{timestamp}\n{SECRET}'
    hmac_code = hmac.new(secret_enc, string_to_sign.encode('utf-8'), digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return f'https://oapi.dingtalk.com/robot/send?access_token={ACCESS_TOKEN}&timestamp={timestamp}&sign={sign}'

def send_dingtalk_alert(content):
    url = get_signed_url()
    headers = {'Content-Type': 'application/json'}
    payload = {
        "msgtype": "text",
        "text": {"content": content},
        "at": {
            "atMobiles": AT_MOBILES,
            "isAtAll": IS_AT_ALL
        }
    }
    try:
        r = requests.post(url, json=payload, headers=headers)
        print(f"[钉钉] 发送成功：{r.status_code}")
    except Exception as e:
        print(f"[错误] 发送钉钉失败：{e}")

# ---------- 系统资源获取 ----------
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_gpu_usage():
    if not gpu_available:
        return None
    try:
        handle = nvmlDeviceGetHandleByIndex(0)
        util = nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
    except:
        return None

# ---------- 主监控循环 ----------
def monitor_loop():
    cpu_usage_history = deque(maxlen=MAX_SAMPLES)
    gpu_usage_history = deque(maxlen=MAX_SAMPLES)

    cpu_alert_sent = False
    gpu_alert_sent = False
    cpu_last_alert_time = None
    gpu_last_alert_time = None

    print("✅ 启动监控：每 5 秒采样，判断过去 1 分钟内是否频繁超阈值")

    while True:
        now = datetime.now()

        # -------- CPU --------
        cpu = get_cpu_usage()
        cpu_usage_history.append(cpu)
        high_cpu_count = sum(1 for x in cpu_usage_history if x > CPU_THRESHOLD)
        print(f"[CPU] 当前: {cpu:.2f}%，过去1分钟超过{CPU_THRESHOLD}%次数: {high_cpu_count}")

        if high_cpu_count >= TRIGGER_COUNT:
            if (not cpu_alert_sent) or (now - cpu_last_alert_time).total_seconds() >= ALERT_INTERVAL:
                send_dingtalk_alert(f"🚨 CPU 使用率过去 1 分钟内超过 {CPU_THRESHOLD}% 的次数达到 {high_cpu_count} 次")
                cpu_alert_sent = True
                cpu_last_alert_time = now
        else:
            cpu_alert_sent = False
            cpu_last_alert_time = None

        # -------- GPU --------
        if gpu_available:
            gpu = get_gpu_usage()
            if gpu is not None:
                gpu_usage_history.append(gpu)
                high_gpu_count = sum(1 for x in gpu_usage_history if x > GPU_THRESHOLD)
                print(f"[GPU] 当前: {gpu:.2f}%，过去1分钟超过{GPU_THRESHOLD}%次数: {high_gpu_count}")

                if high_gpu_count >= TRIGGER_COUNT:
                    if (not gpu_alert_sent) or (now - gpu_last_alert_time).total_seconds() >= ALERT_INTERVAL:
                        send_dingtalk_alert(f"🚨 GPU 使用率过去 1 分钟内超过 {GPU_THRESHOLD}% 的次数达到 {high_gpu_count} 次")
                        gpu_alert_sent = True
                        gpu_last_alert_time = now
                else:
                    gpu_alert_sent = False
                    gpu_last_alert_time = None

        time.sleep(SAMPLE_INTERVAL)

# ---------- 启动入口 ----------
if __name__ == '__main__':
    monitor_loop()
