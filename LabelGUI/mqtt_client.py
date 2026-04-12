import json
import time
from datetime import datetime
import paho.mqtt.client as mqtt


class MQTTManager:
    def __init__(self):
        self.enabled = False
        self.connected = False

        self.host = ""
        self.port = 1883
        self.username = ""
        self.password = ""
        self.topic_prefix = "droneai"

        self.client_id = f"droneai_gui_{int(time.time())}"
        self._client = None

        self.events = []
        self.max_events = 200

        self.locks = {}
        self.message_handler = None  # app.py can assign a callback

    def configure(self, enabled: bool, host: str, port: int, topic_prefix: str,
                  username: str = "", password: str = ""):
        self.enabled = bool(enabled)
        self.host = (host or "").strip()
        self.port = int(port or 1883)
        self.topic_prefix = (topic_prefix or "droneai").strip()
        self.username = (username or "").strip()
        self.password = (password or "").strip()

    def _topic(self, suffix: str) -> str:
        return f"{self.topic_prefix}/{suffix}".strip("/")

    def connect(self):
        if not self.enabled:
            return False, "MQTT is disabled."
        if not self.host:
            return False, "MQTT host is empty."
    
        # Clean up any previous client first
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
    
        self._client = mqtt.Client(client_id=self.client_id, clean_session=True)
    
        if self.username:
            self._client.username_pw_set(self.username, self.password)
    
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
    
        try:
            self._client.connect(self.host, self.port, keepalive=30)
            self._client.loop_start()
            return True, "Connecting..."
        except Exception as e:
            self.connected = False
            return False, f"MQTT connect failed: {e}"

    def disconnect(self):
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
        self.connected = False
        return True, "Disconnected."

    def _on_connect(self, client, userdata, flags, rc):
        self.connected = (rc == 0)
        if self.connected:
            client.subscribe(self._topic("events/#"))
            client.subscribe(self._topic("locks/#"))
            self._push_event({"type": "system", "msg": "MQTT connected", "ts": self._now()})
        else:
            self._push_event({"type": "system", "msg": f"MQTT connect failed rc={rc}", "ts": self._now()})

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        self._push_event({"type": "system", "msg": "MQTT disconnected", "ts": self._now()})

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", errors="ignore")
            data = json.loads(payload)
        except Exception:
            return

        topic = msg.topic or ""
        if "/locks/" in topic:
            lock_key = data.get("lock_key")
            if lock_key:
                self.locks[lock_key] = data
        else:
            self._push_event(data)
            if self.message_handler:
                try:
                    self.message_handler(data)
                except Exception as e:
                    self._push_event({"type": "system", "msg": f"MQTT handler error: {e}", "ts": self._now()})

    def _now(self):
        return datetime.utcnow().isoformat()

    def _push_event(self, e: dict):
        self.events.append(e)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def publish_event(self, event_type: str, payload: dict):
        if not (self.enabled and self.connected and self._client):
            return False
        data = {"type": event_type, "ts": self._now(), **(payload or {})}
        try:
            self._client.publish(self._topic("events/activity"), json.dumps(data), qos=0, retain=False)
            self._push_event(data)
            return True
        except Exception:
            return False

    def publish_lock(self, lock_key: str, by_user: str, status: str):
        if not (self.enabled and self.connected and self._client):
            return False
        data = {
            "lock_key": lock_key,
            "by": by_user,
            "status": status,
            "ts": self._now()
        }
        try:
            self._client.publish(self._topic(f"locks/{lock_key}"), json.dumps(data), qos=0, retain=True)
            self.locks[lock_key] = data
            return True
        except Exception:
            return False
