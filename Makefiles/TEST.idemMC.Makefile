##===- TEST.idem.Makefile ------------------------------*- Makefile -*-===##
#
# Usage: 
#     make TEST=idemGA (detailed list with time passes, etc.)
#     make TEST=idemGA report
#     make TEST=idemGA report.html
#
##===----------------------------------------------------------------------===##

CURDIR  := $(shell cd .; pwd)
PROGDIR := $(PROJ_SRC_ROOT)
RELDIR  := $(subst $(PROGDIR),,$(CURDIR))


$(PROGRAMS_TO_TEST:%=test.$(TEST).%): \
test.$(TEST).%: Output/%.$(TEST).report.txt
	@cat $<

$(PROGRAMS_TO_TEST:%=Output/%.$(TEST).report.txt):  \
Output/%.$(TEST).report.txt: Output/%.linked.rbc $(LOPT) \
	$(PROJ_SRC_ROOT)/TEST.idemGA.Makefile 
	$(VERB) $(RM) -f $@
	@echo "---------------------------------------------------------------" >> $@
	@echo ">>> ========= '$(RELDIR)/$*' Program" >> $@
	@echo "---------------------------------------------------------------" >> $@
	@-$(LLC) -idempotence-construction=size -idempotence-MC=true -idempotence-mc-num-runs=5 -idempotence-mc-move-choose-policy=3 -idempotence-mc-tiebreak-policy=2 -stats -time-passes $< -o $<.s 2>>$@

REPORT_DEPENDENCIES := $(LOPT)
