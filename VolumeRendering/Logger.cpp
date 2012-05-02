#include <stdio.h>
#include <stdarg.h>
#include <time.h>

#include "Logger.h"

static FILE *log_file = NULL;

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

	char date_string[80];
	time_t t;
	t = time(NULL);
	strftime(date_string, sizeof(date_string) - 1, "%a %b %d %Y %H:%M:%S", localtime(&t));

	log("==========================================================================\n");
	log("VolR - Volume rendering engine using CUDA, by Miroslav Beno, STU FIIT 2012\n");
    log("%s \n\n", date_string);
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
	if (log_file != NULL)
		fclose(log_file);
}

