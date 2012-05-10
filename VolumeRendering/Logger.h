/****************************************/
// Simple logger
/****************************************/

#ifndef _LOGGER_H_
#define _LOGGER_H_

class Logger {
	public:
		static void init(char *log_file_name, char mode);
		static void log_time();
		static void log(const char *fmt, ...);
		static void close();
};

#endif