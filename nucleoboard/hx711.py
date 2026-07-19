from machine import Pin
import time


class HX711:
    def __init__(self, dout, sck, gain=128):
        """
        HX711 driver (MicroPython STM32 safe version)

        dout: Pin object (DAT)
        sck: Pin object (CLK)
        """

        self.dout = dout
        self.sck = sck

        self.offset = 0
        self.scale = 1

        # DO NOT touch pins aggressively in init (important for STM32 firmware) #This is AI advice. 
        self.sck.value(0)

    def read_raw(self):

        # wait until ready
        timeout = 100000
        while self.dout.value() == 1:
            timeout -= 1
            if timeout <= 0:
                raise Exception("HX711 timeout (no data ready)")
            time.sleep_us(10)

        value = 0

        # read 24 bits
        for i in range(24):
            self.sck.value(1)
            time.sleep_us(1)

            value = value << 1

            self.sck.value(0)
            time.sleep_us(1)

            if self.dout.value():
                value += 1

        # gain pulse (128 gain)
        self.sck.value(1)
        time.sleep_us(1)
        self.sck.value(0)
        time.sleep_us(1)

        # convert signed
        if value & 0x800000:
            value -= 1 << 24

        return value

    # Averaged reading
    def read(self, samples=5):
        total = 0
        for _ in range(samples):
            total += self.read_raw()
        return total // samples

    # taring
    def tare(self, samples=10):
        self.offset = self.read(samples)


    # Offset corrected value
    def get_value(self, samples=5):
        return self.read(samples) - self.offset

    # Optional scaling (Newton/grams later)
    def set_scale(self, scale):
        self.scale = scale

    def get_units(self, samples=5):
        return (self.get_value(samples) / self.scale)