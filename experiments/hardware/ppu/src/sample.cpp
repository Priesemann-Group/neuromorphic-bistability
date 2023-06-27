#include <array>
#include <stdint.h>
#include <s2pp.h>
#include "libnux/dls.h"
#include "libnux/time.h"

enum Command {RUN, NONE, HALT};
volatile Command command = NONE;

alignas(4) volatile uint8_t parameter = 5;
alignas(4) volatile uint8_t ppu = 1;

int start(void) {
	__vector uint8_t parameters = vec_splat_u8(parameter);
	__vector uint8_t weights = vec_splat_u8(0);
	__vector uint8_t zeros = vec_splat_u8(0);
	
	while(command != HALT) {
		if(command == RUN) {
			command = NONE;

			// update weights
			for(size_t half = 0; half < dls_num_vectors_per_row; half++) {
				for(size_t row = 0; row < dls_num_rows; row++) {
					asm volatile(
						// load and shift weights
						"fxvinx %[weights], %[weight_base], %[row]\n"
						"fxvshb %[weights], %[weights], 1\n"
						// apply updates
						"fxvaddbfs %[weights], %[weights], %[parameters]\n"
						// set to zero if result is smaller than zero
						"fxvcmpb %[weights]\n"
						"fxvsel %[weights], %[weights], %[zeros], 2\n"
						// calculate weight update
						"fxvshb %[weights], %[weights], -1\n"
						"fxvoutx %[weights], %[weight_base], %[row]\n"
						"sync"
						: [weights] "=&qv" (weights)
						: [parameters] "qv" (parameters),
						  [zeros] "qv" (zeros),
						  [weight_base] "b" (dls_weight_base),
						  [row] "r" (row*2 + half)
					);
				}
			}
		}
	}

	return 0;
}
