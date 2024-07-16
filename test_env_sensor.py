import smbus2
import qwiic_bme280
import time
import requests
import json

# Initialize the BME280 sensor
bme280 = qwiic_bme280.QwiicBme280()
if not bme280.is_connected():
    print("The Qwiic BME280 device isn't connected to the system. Please check your connection.")
    exit()

bme280.begin()

# Initialize the I2C bus for ENS160
bus = smbus2.SMBus(1)
ENS160_I2C_ADDRESS = 0x53  # ENS160 default I2C address

# ENS160 Register Addresses
PART_ID_REG = 0x00
OPMODE_REG = 0x10
DATA_AQI_REG = 0x21
DATA_TVOC_REG = 0x22
DATA_ECO2_REG = 0x24
DEVICE_STATUS_REG = 0x20

# Read and print the PART ID
part_id = bus.read_i2c_block_data(ENS160_I2C_ADDRESS, PART_ID_REG, 2)
print(f"ENS160 Part ID: {part_id}")

# Initialize the ENS160 sensor
def initialize_ens160():
    try:
        # Set operating mode to Standard mode
        bus.write_byte_data(ENS160_I2C_ADDRESS, OPMODE_REG, 0x02)
        time.sleep(1)  # Delay to allow sensor to initialize
    except Exception as e:
        print(f"Error initializing ENS160: {e}")

# Read data from the ENS160 sensor
def read_ens160_data():
    try:
        # Check device status
        device_status = bus.read_byte_data(ENS160_I2C_ADDRESS, DEVICE_STATUS_REG)
        print(f"ENS160 Device Status: {device_status}")

        # Read Air Quality Index (AQI)
        air_quality_index = bus.read_byte_data(ENS160_I2C_ADDRESS, DATA_AQI_REG)
        
        # Read TVOC concentration (ppb)
        tvoc_low = bus.read_byte_data(ENS160_I2C_ADDRESS, DATA_TVOC_REG)
        tvoc_high = bus.read_byte_data(ENS160_I2C_ADDRESS, DATA_TVOC_REG + 1)
        tvoc = (tvoc_high << 8) | tvoc_low
        
        # Read Equivalent CO2 concentration (ppm)
        eco2_low = bus.read_byte_data(ENS160_I2C_ADDRESS, DATA_ECO2_REG)
        eco2_high = bus.read_byte_data(ENS160_I2C_ADDRESS, DATA_ECO2_REG + 1)
        eco2 = (eco2_high << 8) | eco2_low

        return air_quality_index, tvoc, eco2
    except Exception as e:
        print(f"Error reading from ENS160: {e}")
        return None, None, None

# Initialize the ENS160 sensor
initialize_ens160()

def send_data(aqi, temperature, tvoc, co2, humidity, pressure):
    url = "http://localhost:5001/environmental-data"
    data = {"aqi": aqi, "temperature": temperature, "tvoc": tvoc, "co": co2, "humidity": humidity, "pressure": pressure}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)

while True:
    # Read data from BME280
    temperature = bme280.temperature_celsius
    humidity = bme280.humidity
    pressure = bme280.pressure

    # Read data from ENS160
    air_quality_index, tvoc, eco2 = read_ens160_data()
    send_data(air_quality_index, temperature, tvoc, eco2, humidity, pressure)

    if air_quality_index is not None:
        # Print ENS160 data
        print(f"ENS160 - Air Quality Index (AQI): {air_quality_index}")
        print(f"ENS160 - Total Volatile Organic Compounds (TVOC): {tvoc} ppb")
        print(f"ENS160 - Estimated CO2 (eCO2): {eco2} ppm")
    
    # Print BME280 data
    print(f"BME280 - Temperature: {temperature:.2f} Â°C")
    print(f"BME280 - Humidity: {humidity:.2f} %")
    print(f"BME280 - Pressure: {pressure:.2f} hPa")

    time.sleep(2)