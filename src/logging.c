#include "logging.h"
#include "macro/constant.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void logger(const log_tag tag, const char *message)
{
   time_t now;
   time(&now);
   switch (tag)
   {
   case INFO:
      printf(ANSI_COLOR_GREEN "-- %s [INFO]: %s\n" ANSI_COLOR_RESET, ctime(&now), message);
      break;
   case WARN:
      printf(ANSI_COLOR_YELLOW "-- %s [WARN]: %s\n" ANSI_COLOR_RESET, ctime(&now), message);
      break;

   case FATAL:
      printf(ANSI_COLOR_RED "-- %s [FATAL]: %s\n" ANSI_COLOR_RESET, ctime(&now), message);
      break;

   default:
      break;
   }
}