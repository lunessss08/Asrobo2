"""maze: errors."""



class Aborted(Exception):
    """ ผู้ใช้กด STOP / QUIT ระหว่างที่หุ่นทำงาน """


class HardwareFault(Exception):
    """ เซ็นเซอร์หรือการเคลื่อนที่ผิดปกติจนทำต่อไม่ปลอดภัย (คนละอย่างกับผู้ใช้กดหยุด) """


