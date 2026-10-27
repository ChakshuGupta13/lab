APPLICATION = mlkem_unfaulted
BOARD ?= nrf52840dk
RIOTBASE ?= $(CURDIR)/../riot/RIOT
MLKEM_DIR ?= $(CURDIR)/mlkem

USEMODULE += stdio_uart_rx
INCLUDES += -I$(MLKEM_DIR)
CFLAGS += -DTHREAD_STACKSIZE_MAIN=32768

include $(RIOTBASE)/Makefile.include