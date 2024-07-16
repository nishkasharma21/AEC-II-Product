import board
import time
import busio
import adafruit_bme680
import adafruit_ccs811
import requests
import json

# Initialize I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize BME680 sensor
bme680 = adafruit_bme680.Adafruit_BME680_I2C(i2c)

# Initialize CCS811 sensor
ccs811 = adafruit_ccs811.CCS811(i2c)

def calculate_aqi(co2, tvoc, humidity):
    """
    Calculate the Air Quality Index (AQI) based on CO2, TVOC, and humidity levels.

    :param co2: CO2 level in ppm
    :param tvoc: TVOC level in ppb
    :param humidity: Humidity level in %
    :return: AQI value (0-500)
    """
    # Define AQI categories and corresponding pollutant concentrations
    aqi_categories = [
        {"name": "Good", "co2": (0, 450), "tvoc": (0, 220), "humidity": (30, 60)},
        {"name": "Fair", "co2": (451, 900), "tvoc": (221, 660), "humidity": (20, 30) or (60, 80)},
        {"name": "Poor", "co2": (901, 1200), "tvoc": (661, 1320), "humidity": (10, 20) or (80, 90)},
        {"name": "Hazardous", "co2": (1201, 2000), "tvoc": (1321, 2200), "humidity": (0, 10) or (90, 100)},
    ]

    # Calculate AQI based on pollutant concentrations
    aqi = 0
    for category in aqi_categories:
        if (category["co2"][0] <= co2 <= category["co2"][1] and
            category["tvoc"][0] <= tvoc <= category["tvoc"][1] and
            (category["humidity"][0] <= humidity <= category["humidity"][1] or
             (isinstance(category["humidity"], tuple) and
              category["humidity"][0] <= humidity <= category["humidity"][1]))):
            aqi = aqi_categories.index(category) * 125
            break

    return aqi

def send_data(aqi, temperature, tvoc, co2, humidity, pressure):
    # Send data to Mac Mini using HTTP
    url = "http://localhost:5001/environmental-data"
    data = {"aqi": aqi, "temperature": temperature, "tvoc": tvoc, "co": co2, "humidity": humidity, "pressure": pressure}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)

previous_data = None

while True:
    # Read data from BME680 sensor
    humidity = bme680.humidity
    pressure = bme680.pressure

    # Read data from CCS811 sensor
    co2 = ccs811.eco2
    tvoc = ccs811.tvoc
    temperature = ccs811.temperature

    aqi = calculate_aqi(co2, tvoc, humidity)

    current_data = (aqi, temperature, tvoc, co2, humidity, pressure)

    if current_data!= previous_data:
        send_data(*current_data)
        previous_data = current_data

    # Wait for 1 minute before sending next data
    time.sleep(1)