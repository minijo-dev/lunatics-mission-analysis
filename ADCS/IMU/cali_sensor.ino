#include <Adafruit_BNO08x.h>

#define BNO08X_RESET -1
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

float ax, ay, az, gx, gy, gz, mx, my, mz;

void setup(void) {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("Adafruit BNO08x - Calibrated Sensor Readings");

  if (!bno08x.begin_I2C()) {
    Serial.println("Failed to find BNO08x chip");
    while (1) delay(10);
  }

  Serial.println("BNO08x Found!");
  setReports();
  delay(100);
}

void setReports(void) {
  Serial.println("Setting calibrated sensor reports");

  if (!bno08x.enableReport(SH2_ACCELEROMETER, 10000)) {
    Serial.println("Could not enable calibrated accelerometer");
  }
  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000)) {
    Serial.println("Could not enable calibrated gyroscope");
  }
  if (!bno08x.enableReport(SH2_MAGNETIC_FIELD_CALIBRATED, 10000)) {
    Serial.println("Could not enable calibrated magnetometer");
  }
}

void loop() {
  while (bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ACCELEROMETER:
        ax = sensorValue.un.accelerometer.x;
        ay = sensorValue.un.accelerometer.y;
        az = sensorValue.un.accelerometer.z;
        break;

      case SH2_GYROSCOPE_CALIBRATED:
        gx = sensorValue.un.gyroscope.x;
        gy = sensorValue.un.gyroscope.y;
        gz = sensorValue.un.gyroscope.z;
        break;

      case SH2_MAGNETIC_FIELD_CALIBRATED:
        mx = sensorValue.un.magneticField.x;
        my = sensorValue.un.magneticField.y;
        mz = sensorValue.un.magneticField.z;
        break;
    }
  }

  // Print all values together after processing available events
  Serial.print(sensorValue.status); Serial.print(", ");
  Serial.print(ax); Serial.print(", ");
  Serial.print(ay); Serial.print(", ");
  Serial.print(az); Serial.print(", ");
  Serial.print(gx); Serial.print(", ");
  Serial.print(gy); Serial.print(", ");
  Serial.print(gz); Serial.print(", ");
  Serial.print(mx); Serial.print(", ");
  Serial.print(my); Serial.print(", ");
  Serial.println(mz);

  delay(10); // match the 100 Hz sample rate
}
