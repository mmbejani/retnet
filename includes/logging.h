#pragma once

typedef enum
{
    INFO,
    WARN,
    FATAL
} log_tag;

void logger(const log_tag tag, const char *message);