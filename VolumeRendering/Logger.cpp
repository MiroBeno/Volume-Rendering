#include <stdio.h>
#include <stdarg.h>
#include <time.h>

#include "Logger.h"

static FILE *log_file = NULL;
static time_t start_time = 0, last_time = 0;

void Logger::init(char *log_file_name, char mode) {
	printf("Initializing logger...\n");
	switch(mode) {
		case 'a':											// append to log file
			log_file = fopen(log_file_name, "a");
			break;
		case 'w':											// overwrite log file
			log_file = fopen(log_file_name, "w");
			break;
		case 'n':
			log_file = NULL;
			break;
	}
	if (log_file == NULL) 
		printf("Logging to file disabled\n");
	else {
		printf("Logging intialized: %s\n", log_file_name);
	}

	log("==========================================================================\n");
	log_time();
	start_time = last_time;
	log("VolR - Volume rendering engine using CUDA, by Miroslav Beno, STU FIIT 2012\n");
    log("\n");
}

void Logger::log_time() {
	char date_string[80];
	time_t t = time(NULL);
	strftime(date_string, sizeof(date_string) - 1, "%a %b %d %Y %H:%M:%S", localtime(&t));
    log("%s\n", date_string);
	last_time = t;
}

void Logger::log(const char *fmt, ...) {
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	if(log_file != NULL)
		vfprintf(log_file, fmt, args);
	va_end(args);
	if (log_file != NULL)
		fflush(log_file);
}

void Logger::close() {
	log_time();
	if (start_time != 0) {
		int seconds = (int) difftime(last_time, start_time);
		log("Application run time: %im %is\n", seconds / 60, seconds % 60);
	}
	log("==========================================================================\n");
	if (log_file != NULL)
		fclose(log_file);
}

