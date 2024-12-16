---
jupyter:
  kernelspec:
    display_name: base
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.4
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .code execution_count="31"}
``` python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import random
```
:::

::: {.cell .code execution_count="32"}
``` python
def generate_pn_code(length):
    """Generate PN code with values 0 and 1"""
    return np.random.choice([0, 1], size=length)

# Generate 32 PN codes, each 1024 bits long
pn_codes = [generate_pn_code(1024) for _ in range(32)]


pn_code_x = pn_codes[0]
pn_code_y = pn_codes[1]

plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(pn_code_x)
plt.title("PN Code 1")
plt.xlabel("Bit Index")
plt.ylabel("Value")

plt.subplot(2, 1, 2)
plt.plot(pn_code_y)
plt.title("PN Code 2")
plt.xlabel("Bit Index")
plt.ylabel("Value")

plt.tight_layout()
plt.show()



# Define the range to zoom in on
zoom_range = 100

# Plot the zoomed-in signals
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(pn_code_x[:zoom_range])
plt.title("Zoomed-in PN Code 1")
plt.xlabel("Bit Index")
plt.ylabel("Value")

plt.subplot(2, 1, 2)
plt.plot(pn_code_y[:zoom_range])
plt.title("Zoomed-in PN Code 2")
plt.xlabel("Bit Index")
plt.ylabel("Value")

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_8766471c28844445b54af30cbae16292/12d23eb7ed5f22173ac0e848792f25e6deb1b450.png)
:::

::: {.output .display_data}
![](vertopal_8766471c28844445b54af30cbae16292/470894e5dc030eb28353ff15fbdab5a4c6c358d2.png)
:::
:::

::: {.cell .code execution_count="33"}
``` python
def bpsk_modulate(pn_code):
    """Modulate a PN code using BPSK, converting 0 to -1 and 1 to 1"""
    return np.where(np.array(pn_code) == 1, 1, -1)

# Modulate all 32 PN codes
bpsk_modulated_signals = [bpsk_modulate(code) for code in pn_codes]

# Plot the modulated signals for the first two PN codes (zoomed-in view for clarity)
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(bpsk_modulated_signals[0][:zoom_range])
plt.title("BPSK Modulated PN Code 1 (Zoomed-in)")
plt.xlabel("Bit Index")
plt.ylabel("Value")

plt.subplot(2, 1, 2)
plt.plot(bpsk_modulated_signals[1][:zoom_range])
plt.title("BPSK Modulated PN Code 2 (Zoomed-in)")
plt.xlabel("Bit Index")
plt.ylabel("Value")

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_8766471c28844445b54af30cbae16292/1d09a36a3b6407bedf7d2e5e9b67f0505c16f411.png)
:::
:::

::: {.cell .code execution_count="34"}
``` python
# Parameters
fs = 1024  # Sampling frequency
A_n = 0.5  # Noise amplitude

# Time vector for plotting (assuming bit duration is 1 sample)
t = np.arange(0, fs)

# Signal to transmit (one of the BPSK modulated signals)
x = bpsk_modulated_signals[0]

# Additive noise
noise = A_n * np.random.normal(size=fs)
noisy_x = x + noise

# Plotting the noisy signal
plt.figure(figsize=(10, 6))
plt.plot(t, noisy_x)
plt.title("Noisy BPSK Signal")
plt.xlabel("Bit Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
```

::: {.output .display_data}
![](vertopal_8766471c28844445b54af30cbae16292/a3a853eee8e2733b21f575648f7e88d9bf9d36b3.png)
:::
:::

::: {.cell .code execution_count="35"}
``` python
def correlate(received_signal, pn_code):
    """Compute the correlation between the received signal and a PN code"""
    modulated_pn_code = bpsk_modulate(pn_code)
    correlation = np.correlate(received_signal, modulated_pn_code, mode='full')
    return correlation

# Find the best matching PN code
correlations = [correlate(noisy_x, code) for code in pn_codes]
best_match_index = np.argmax([np.max(c) for c in correlations])
best_match_code = pn_codes[best_match_index]

print(f"Best match index: {best_match_index}")
print(f"Highest correlation value: {np.max(correlations[best_match_index])}")

# Decode the entire received signal using the best matching PN code
best_correlation = correlations[best_match_index]
decoded_signal = np.sign(best_correlation)  # Simplified example of decoding

# Correlation plot for the best matching PN code
plt.figure(figsize=(10, 6))
plt.plot(best_correlation)
plt.title("Correlation with Best Matching PN Code")
plt.xlabel("Sample Index")
plt.ylabel("Correlation Value")
plt.grid()
plt.show()

# Plot the decoded signal (zoomed-in view for clarity)
plt.figure(figsize=(10, 6))
plt.plot(decoded_signal[:zoom_range])
plt.title("Decoded Signal (Zoomed-in)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
```

::: {.output .stream .stdout}
    Best match index: 0
    Highest correlation value: 1019.1604116749845
:::

::: {.output .display_data}
![](vertopal_8766471c28844445b54af30cbae16292/8fd205702b832e3872f8ad39b314c3f921f7aae0.png)
:::

::: {.output .display_data}
![](vertopal_8766471c28844445b54af30cbae16292/fb1fce18cc247d84217c33097de028f2cf85405a.png)
:::
:::

::: {.cell .code execution_count="36"}
``` python
# Select three random PN codes (simulated satellites)
random_satellites_indices = random.sample(range(32), 3)
random_satellites = [pn_codes[i] for i in random_satellites_indices]

print("Selected Satellites Indices for Location Calculation:", random_satellites_indices)
```

::: {.output .stream .stdout}
    Selected Satellites Indices for Location Calculation: [26, 2, 21]
:::
:::

::: {.cell .code execution_count="37"}
``` python
def calculate_location(sat_signals):
    return [sum(signal) for signal in sat_signals]  # Simplified example

location = calculate_location(random_satellites)
print("Simulated Location (Latitude, Longitude):", location)
```

::: {.output .stream .stdout}
    Simulated Location (Latitude, Longitude): [485, 507, 515]
:::
:::

::: {.cell .code execution_count="38"}
``` python
# Select four random PN codes (simulated satellites)
random_altitude_sat_indices = random.sample(range(32), 4)
random_altitude_sats = [pn_codes[i] for i in random_altitude_sat_indices]

print("Selected Satellites Indices for Altitude Calculation:", random_altitude_sat_indices)
```

::: {.output .stream .stdout}
    Selected Satellites Indices for Altitude Calculation: [5, 2, 9, 21]
:::
:::

::: {.cell .code execution_count="39"}
``` python
# Dummy function to simulate altitude calculation
def calculate_altitude(sat_signals):
    """Calculate altitude based on satellite signals (dummy implementation)"""
    return sum([sum(signal) for signal in sat_signals]) / len(sat_signals)  # Simplified example

# Calculate altitude using the selected satellites
altitude = calculate_altitude(random_altitude_sats)
print("Simulated Altitude:", altitude)
```

::: {.output .stream .stdout}
    Simulated Altitude: 508.25
:::
:::

::: {.cell .code}
``` python
```
:::
