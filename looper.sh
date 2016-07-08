sh synclogs >> /tmp/synclogs.log
echo '\nsynced\n'
python get_status.py > /tmp/cur_status
echo '\n-------------------' `date` $'---------------\n'
cat /tmp/cur_status
