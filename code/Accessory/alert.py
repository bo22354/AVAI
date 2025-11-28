import requests

# PASTE YOUR WEBHOOK URL HERE
WEBHOOK_URL = "https://discord.com/api/webhooks/1442890378238758953/xeGCzzEmXtG0XPWtZ2jiA6vfp0y0SYgggsTu-pcBXfJt-bwXRxnfimY4WkOi6TH9_bXL"

def send_loud_notification(message="Job Done"):
    data = {
        "content": message,
        "tts": True  # <--- THIS IS THE MAGIC SWITCH
    }
    
    try:
        requests.post(WEBHOOK_URL, json=data)
        print("Loud notification sent.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    send_loud_notification("Training Complete! Check results !!!")