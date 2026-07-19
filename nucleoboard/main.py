from machine import Pin
from biorobotics import PWM, Encoder
from hx711 import HX711
import time


####### HX711 pins #########


time.sleep(1)

DAT = Pin("D9", Pin.IN, Pin.PULL_UP)
CLK = Pin("D8", Pin.OUT)

hx = HX711(DAT, CLK)

print("Taring...")
time.sleep(2)

hx.tare()


########## Motor driver pins##############

DIR_PIN = "D4"
BRAKE_PIN = "D7"
PWM_PIN = "D5"

ENCODER_A = "D12"
ENCODER_B = "D11"



########## Motion settings ############

HIT_POSITION = 800
HOME_POSITION = 0

# Time between cycles
CYCLE_TIME = 3000


############## PD controller ##################

KP = 0.13
KD = 0.10


MAX_SPEED = 100
MIN_SPEED = 25


POSITION_TOLERANCE = 70



################ Controller timing ####################

CONTROL_INTERVAL = 8       # ms

MOVE_TIMEOUT = 2500       # increased safety margin

STALL_TIMEOUT = 500        # less aggressive



###### Force sampling ########

PRINT_INTERVAL_MS = 150
FORCE_SAMPLE_INTERVAL_MS = 150



######### Safety limits ##########
# (chosen way to big right now, mostly usefull for testing)

MIN_POSITION = -5000
MAX_POSITION = 5000



# Encoder direction
# 0 = backwards (your current working direction)
# 1 = forwards

ENCODER_DIRECTION = 0



####### Hardware initialization #######

motor_direction = Pin(
    DIR_PIN,
    Pin.OUT
)

motor_brake = Pin(
    BRAKE_PIN,
    Pin.OUT
)


motor_pwm = PWM(
    PWM_PIN
)


encoder = Encoder(
    ENCODER_A,
    ENCODER_B
)



encoder_zero = encoder.counter()



####### Force variables  ##########

latest_force_raw = None
latest_force = None



############ Logging ######

log_data = []

experiment_start = time.ticks_ms()

LOG_INTERVAL = 20

last_log_time = time.ticks_ms()



def log_sample(position, target, error, speed, direction):

    global latest_force_raw
    global latest_force

    log_data.append(
        (
            time.ticks_diff(
                time.ticks_ms(),
                experiment_start
            ),

            position,
            target,
            error,
            speed,
            direction,
            latest_force_raw,
            latest_force
        )
    )



def save_csv(filename="measurement.csv"):

    print("Saving CSV...")

    with open(filename, "w") as f:

        f.write(
            "time_ms;"
            "position;"
            "target;"
            "error;"
            "speed;"
            "direction;"
            "force_raw;"
            "force_N\n"
        )


        for row in log_data:

            if row[6] is None:
                raw = ""
            else:
                raw = str(row[6])


            if row[7] is None:
                force = ""
            else:
                force = "{:.3f}".format(
                    row[7]
                ).replace(".", ",")


            f.write(
                "{};{};{};{};{};{};{};{}\n".format(
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    raw,
                    force
                )
            )


    print(
        "Saved",
        len(log_data),
        "samples"
    )

##### Utility functions #######


def get_position():

    return (
        encoder.counter()
        -
        encoder_zero
    )



def sample_hx_once():

    try:

        value = hx.get_value()

    except Exception:

        return None


    if value is None:

        return None


    return value



def drive_motor(direction, speed):

    speed = int(abs(speed))


    if speed > MAX_SPEED:

        speed = MAX_SPEED


    # release brake
    motor_brake.value(0)


    motor_direction.value(
        direction
    )


    motor_pwm.write_percentage(
        speed
    )



def stop_motor():

    motor_pwm.write_percentage(0)

    # engage brake
    motor_brake.value(1)




############ Position controller ###########


def move_to_position(target):


    global latest_force_raw
    global latest_force
    global last_log_time



    stop_motor()


    start_time = time.ticks_ms()



    previous_error = (
        target
        -
        get_position()
    )


    last_position = get_position()

    last_change = start_time



    force_sample_last = time.ticks_ms()



    while True:



        ########### Force sampling##########

        if time.ticks_diff(
            time.ticks_ms(),
            force_sample_last
        ) >= FORCE_SAMPLE_INTERVAL_MS:


            value = sample_hx_once()


            latest_force_raw = value


            if value is None:

                latest_force = None


            else:

                try:

                    latest_force = (
                        value - 6482
                    ) / (
                        -186665
                    )


                except Exception:

                    latest_force = None



            force_sample_last = time.ticks_ms()





        ####### Controller timing########


        loop_start = time.ticks_ms()

        current = get_position()

        now = loop_start





        ########## Safety limits##############


        if (
            current < MIN_POSITION
            or
            current > MAX_POSITION
        ):


            stop_motor()


            print(
                "Position limit exceeded:",
                current
            )


            return False





        ############ Error calculation#########


        error = target - current





        ############Target reached###########


        if abs(error) <= POSITION_TOLERANCE:


            stop_motor()


            print(
                "Reached:",
                target,
                "position:",
                current
            )


            return True





        ###### Timeout#######


        if time.ticks_diff(
            now,
            start_time
        ) > MOVE_TIMEOUT:


            stop_motor()


            print(
                "Movement timeout",
                "Current:",
                current
            )


            return False





        ######### Encoder stall detection###########


        if current != last_position:


            last_position = current

            last_change = now



        if time.ticks_diff(
            now,
            last_change
        ) > STALL_TIMEOUT:


            stop_motor()


            print(
                "Encoder stopped changing"
            )


            return False





        ####### PD controller #######


        velocity = (
            error
            -
            previous_error
        )


        output = (
            KP * error
            +
            KD * velocity
        )


        previous_error = error





       ######### Direction #########


        if output > 0:

            direction = ENCODER_DIRECTION

        else:

            direction = 1 - ENCODER_DIRECTION





        ##### Speed ########


        speed = abs(output)


        speed = max(
            speed,
            MIN_SPEED
        )


        speed = min(
            speed,
            MAX_SPEED
        )





        # Slow near home

        if (
            target == HOME_POSITION
            and
            abs(error) < 200
        ):

            speed = min(
                speed,
                20
            )





        ######### Logging #########


        if time.ticks_diff(
            time.ticks_ms(),
            last_log_time
        ) >= LOG_INTERVAL:


            log_sample(
                current,
                target,
                error,
                int(speed),
                direction
            )


            last_log_time = time.ticks_ms()





        ######### Drive motor ######


        drive_motor(
            direction,
            speed
        )





        ##### Maintain control frequency########


        loop_time = time.ticks_diff(
            time.ticks_ms(),
            loop_start
        )


        remaining_time = (
            CONTROL_INTERVAL
            -
            loop_time
        )


        if remaining_time > 0:

            time.sleep_ms(
                remaining_time
            )

######### Movement cycle ###########


last_hit_time = None



def run_cycle():

    global last_hit_time



    print(
        "\nMoving to hit position"
    )



    if not move_to_position(
        HIT_POSITION
    ):

        return False



    current_hit_time = time.ticks_ms()



    if last_hit_time is not None:

        print(
            "Time between hits:",
            time.ticks_diff(
                current_hit_time,
                last_hit_time
            ),
            "ms"
        )

    else:

        print(
            "First hit"
        )



    last_hit_time = current_hit_time





    print(
        "\nMoving to home position"
    )



    if not move_to_position(
        HOME_POSITION
    ):

        return False



    return True





######### Startup ############


stop_motor()



print("-----------------------------")

print(
    "Motor controller started"
)


print(
    "Initial position:",
    get_position()
)


print(
    "Speed range:",
    MIN_SPEED,
    "-",
    MAX_SPEED
)


print(
    "PD gains:",
    "KP:",
    KP,
    "KD:",
    KD
)


print("-----------------------------")





######### Experiment timing ##########

EXPERIMENT_DURATION = 15000   # ms


experiment_end = time.ticks_add(
    time.ticks_ms(),
    EXPERIMENT_DURATION
)



next_cycle = time.ticks_add(
    time.ticks_ms(),
    CYCLE_TIME
)



last_force_print = time.ticks_ms()





########### Main loop ########


while True:


    now = time.ticks_ms()





    ############ Continuous force update ############


    if time.ticks_diff(
        now,
        last_force_print
    ) >= PRINT_INTERVAL_MS:



        value = sample_hx_once()


        latest_force_raw = value



        if value is None:

            latest_force = None


        else:

            try:

                latest_force = (
                    value - 6482
                ) / (
                    -186665
                )


            except Exception:

                latest_force = None



        last_force_print = now





    ######## Stop experiment ########


    if time.ticks_diff(
        now,
        experiment_end
    ) >= 0:


        stop_motor()


        print(
            "Experiment finished"
        )


        save_csv()


        break





    ######### Start cycle ########


    if time.ticks_diff(
        now,
        next_cycle
    ) >= 0:


        print(
            "\n-------------------------------"
        )


        print(
            "Starting cycle"
        )


        print(
            "-------------------------------"
        )



        success = run_cycle()



        if not success:
            stop_motor()
            print("Cycle failed, trying next cycle")
            next_cycle = time.ticks_add(
                time.ticks_ms(),
                CYCLE_TIME
            )
            continue


        # Keep schedule

        next_cycle = time.ticks_add(
            next_cycle,
            CYCLE_TIME
        )





    # Low priority loop

    time.sleep_ms(10)





print(
    "Program stopped. CSV saved."
)


