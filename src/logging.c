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
      printf(ANSI_COLOR_GREEN);
      printf("-- %s [INFO]: %s\n", ctime(&now), message);
      printf(ANSI_COLOR_RESET);
      break;
   case WARN:
      printf(ANSI_COLOR_YELLOW);
      printf("-- %s [WARN]: %s\n", ctime(&now), message);
      print(ANSI_COLOR_RESET);
      break;

   default:
      break;
   }
}