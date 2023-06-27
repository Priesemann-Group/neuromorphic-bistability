#include <array>
#include <stdint.h>
#include <s2pp.h>
#include "libnux/dls.h"
#include "libnux/time.h"

enum Command {RUN, NONE, HALT};
volatile Command command = NONE;

alignas(4) volatile uint8_t lambda = 5;
alignas(4) volatile uint8_t mu = 20;
alignas(4) volatile uint8_t offset = 114;
alignas(4) volatile uint8_t random_seed = 0x1f;
alignas(4) volatile uint8_t degree_seed = 0x3f;

alignas(4) volatile uint8_t ppu = 1;

constexpr uint32_t left_sram_cnt_read = 0x122800;
constexpr uint32_t right_sram_cnt_read = 0x12a800;
constexpr uint32_t left_sram_cnt_reset = 0x122c00;
constexpr uint32_t right_sram_cnt_reset = 0x12ac00;

alignas(4) constexpr std::array<std::array<uint8_t, 16*8>, 2> order = {{
	{{15,  14,  13,  12,  11,  10,   9,   8,   7,   6,   5,   4,   3,
	   2,   1,   0,  47,  46,  45,  44,  43,  42,  41,  40,  39,  38,
	  37,  36,  35,  34,  33,  32,  79,  78,  77,  76,  75,  74,  73,
	  72,  71,  70,  69,  68,  67,  66,  65,  64, 111, 110, 109, 108,
	 107, 106, 105, 104, 103, 102, 101, 100,  99,  98,  97,  96, 143,
	 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130,
	 129, 128, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165,
	 164, 163, 162, 161, 160, 207, 206, 205, 204, 203, 202, 201, 200,
	 199, 198, 197, 196, 195, 194, 193, 192, 239, 238, 237, 236, 235,
	 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224}},
	{{31,  30,  29,  28,  27,  26,  25,  24,  23,  22,  21,  20,  19,
	  18,  17,  16,  63,  62,  61,  60,  59,  58,  57,  56,  55,  54,
	  53,  52,  51,  50,  49,  48,  95,  94,  93,  92,  91,  90,  89,
	  88,  87,  86,  85,  84,  83,  82,  81,  80, 127, 126, 125, 124,
	 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 159,
	 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146,
	 145, 144, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181,
	 180, 179, 178, 177, 176, 223, 222, 221, 220, 219, 218, 217, 216,
	 215, 214, 213, 212, 211, 210, 209, 208, 255, 254, 253, 252, 251,
	 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240}}
}};

void read_count(std::array<std::array<uint8_t, dls_vector_size>, dls_num_vectors_per_row>& rates) {
	uint32_t nrn_addr;
	uint8_t nrn;
	for(size_t i = 0; i < dls_num_vectors_per_row; i++) {
		for(size_t j = 0; j < dls_vector_size; j++) {
			nrn = order[i][j];
			if ((nrn / 128) == 0) {
				nrn_addr = left_sram_cnt_read;
			}
			else {
				nrn_addr = right_sram_cnt_read;
			}
			nrn_addr += 4 * (nrn & 0x7f) + 512 * ppu;
			rates[i][j] = (omnibus_read(nrn_addr) & 0xff) >> 1;
		}
	}
}

void reset_count() {
	uint32_t nrn_addr;
	uint8_t nrn;
	for(size_t i = 0; i < dls_num_vectors_per_row; i++) {
		for(size_t j = 0; j < dls_vector_size; j++) {
			nrn = order[i][j];
			if ((nrn / 128) == 0) {
				nrn_addr = left_sram_cnt_reset;
			}
			else {
				nrn_addr = right_sram_cnt_reset;
			}
			nrn_addr += 4 * (nrn & 0x7f) + 512 * ppu;
			omnibus_write(nrn_addr, 0);
		}
	}
}

void wait_time_us(uint32_t time, uint32_t frequency) {
	sleep_cycles(time * frequency);
}

void set_rand(uint8_t seed, size_t row) {
	
	__vector uint8_t seeds = vec_splat_u8(seed);
	
	asm volatile(
		"fxvoutx %[seeds], %[base], %[row]\n"
		:
		: [seeds] "qv" (seeds),
		  [base] "b" (dls_randgen_base),
		  [row] "r" (row)
		:
	);
}

int start(void) {
	uint32_t random_row = 0;
	uint32_t degree_row = 1;	

	set_rand(random_seed, random_row);

	uint32_t start;
	uint32_t measurement_duration = 1000;
	uint32_t duration;
	std::array<std::array<uint8_t, 128>, 2> rates_full;

	__vector uint8_t lambdas = vec_splat_u8(lambda);
	__vector uint8_t targets = vec_splat_u8(mu);
	__vector uint8_t offsets = vec_splat_u8(offset);
	__vector uint8_t scales = vec_splat_u8(10);
	__vector uint8_t zeros = vec_splat_u8(0);
	__vector uint8_t rates = vec_splat_u8(0);
	__vector uint8_t weights = vec_splat_u8(0);
	__vector uint8_t randoms = vec_splat_u8(0);
	__vector uint8_t updates = vec_splat_u8(0);
	__vector uint8_t degrees = vec_splat_u8(0);
	
	while(command != HALT) {
		if(command == RUN) {
			command = NONE;

			// same sparsity seed for every run
			set_rand(degree_seed, degree_row);
			
			// ensure that omnibus read does not take significantly longer
			duration = UINT32_MAX;
			while(duration > (measurement_duration * 250 + 250000)) {
				start = get_time_base();
			
				// reset rate counter
				reset_count();
				
				wait_time_us(measurement_duration, 250);
				
				// read rates
				read_count(rates_full);
	
				duration = get_time_base() - start;
			}
			
			// update weights
			for(size_t half = 0; half < dls_num_vectors_per_row; half++) {
				for(size_t i = 0; i < dls_vector_size; i++) {
					rates[i] = rates_full[half][i];
				}

				for(size_t row = 0; row < dls_num_rows; row++) {
					asm volatile(
						// calculate rate update
						"fxvsubbfs %[updates], %[targets], %[rates]\n"
						"fxvmulbfs %[updates], %[updates], %[lambdas]\n"
						// calculate random update
						"fxvinx %[randoms], %[rand_base], %[random_row]\n"
						"fxvmulbfs %[randoms], %[randoms], %[scales]\n"
						// load and shift weights
						"fxvinx %[weights], %[weight_base], %[row]\n"
						"fxvshb %[weights], %[weights], 1\n"
						// apply updates
						"fxvaddbfs %[updates], %[updates], %[randoms]\n"
						"fxvaddbfs %[weights], %[weights], %[updates]\n"
						// set to zero if result is smaller than zero
						"fxvcmpb %[weights]\n"
						"fxvsel %[weights], %[weights], %[zeros], 2\n"
						// generate sparsity
						"fxvinx %[degrees], %[rand_base], %[degree_row]\n"
						"fxvsubbfs %[degrees], %[degrees], %[offsets]\n"
						"fxvcmpb %[degrees]\n"
						"fxvsel %[weights], %[weights], %[zeros], 2\n"
						// calculate weight update
						"fxvshb %[weights], %[weights], -1\n"
						"fxvoutx %[weights], %[weight_base], %[row]\n"
						"sync"
						: [weights] "=&qv" (weights),
						  [randoms] "=&qv" (randoms),
						  [updates] "=&qv" (updates),
						  [degrees] "=&qv" (degrees)
						: [lambdas] "qv" (lambdas),
						  [targets] "qv" (targets),
						  [scales] "qv" (scales),
						  [rates] "qv" (rates),
						  [zeros] "qv" (zeros),
						  [offsets] "qv" (offsets),
						  [weight_base] "b" (dls_weight_base),
						  [rand_base] "b" (dls_randgen_base),
						  [row] "r" (row*2 + half),
						  [random_row] "r" (random_row),
						  [degree_row] "r" (degree_row)
					);
				}
			}
		}
	}

	return 0;
}
