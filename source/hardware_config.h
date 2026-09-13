#ifndef HARDWARE_CONFIG_H_
#define HARDWARE_CONFIG_H_

// Verify the optical assembly LED GPIO against the board wiring before
// enabling it. The example pin is intentionally unused by default.
#define MICROPLASTIC_LED_CONTROL_ENABLED 0
#define MICROPLASTIC_LED_GPIO_BASE GPIO0
#define MICROPLASTIC_LED_GPIO_PIN 10U

// Existing project mapping: ADC0 channel 2 / ADC0_A2 / J8 pin 12.
#define MICROPLASTIC_ADC_CHANNEL 2U

#endif  // HARDWARE_CONFIG_H_
