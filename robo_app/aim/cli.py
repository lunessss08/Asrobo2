"""aim: cli."""

from .app import AutoAimApp
import argparse

def main():
    parser = argparse.ArgumentParser(description="RoboMaster EP ยิงเป้าตามลำดับสีอัตโนมัติ")
    parser.add_argument("--calibrate", action="store_true",
                        help="เริ่มที่หน้า calibrate (วาดพื้นที่ตาบอด + จูนสีครบทุกสี) รันครั้งแรกจะทำให้อัตโนมัติ")
    AutoAimApp(calibrate=parser.parse_args().calibrate).run()
