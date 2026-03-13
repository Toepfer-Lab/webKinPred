.PHONY: clean-artifacts clean-pycache

clean-artifacts:
	rm -rf db.sqlite3 celerybeat-schedule
	rm -rf media staticfiles mmseqs_tmp
	rm -rf fastas/dbs
	rm -rf /tmp/webkinpred_*
	$(MAKE) clean-pycache

clean-pycache:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
