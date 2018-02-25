path$ = "D:\tmp\"
out$ = path$+"formants-5.txt"

writeFileLine: out$, "ID_rec"+tab$+"ID_vow"+tab$+"label_wov"+tab$+"F1_1"+tab$+"F1_2"+tab$+"F1_3"+tab$+"F1_4"+tab$+"F1_5"+tab$+"F2_1"+tab$+"F2_2"+tab$+"F2_3"+tab$+"F2_4"+tab$+"F2_5"+tab$+"include"

#set items ID

select all
total=numberOfSelected("TextGrid")
for i from 1 to 'total'
	textID'i' = selected ("TextGrid",i)
	formantID'i' = selected ("Formant",i)
endfor

for i from 1 to 'total'
	textID = textID'i'
	formantID = formantID'i'
	select 'formantID'
	name$ = selected$ ("Formant")
	select 'textID'
	intv = Get number of intervals... 4

	# iterate through intervals
	for j from 1 to 'intv'
		select 'textID'
		lbl$ = Get label of interval... 4 j

		if ((lbl$ <> "") and (lbl$ <> " "))
			start = Get starting point... 4 j
			end = Get end point... 4 j
            tmp = Get interval at time: 3, start
            vow$ = Get label of interval: 3, tmp
            if index(vow$,"*")
                incl = 0
            else
                incl = 1
            endif

            step = (end - start)/6
            select 'formantID'
            f1_str$ = ""
            f2_str$ = ""
            for k from 1 to 5
                time = start + k*step
                f1 = Get value at time: 1, time, "Hertz", "Linear"
                f2 = Get value at time: 2, time, "Hertz", "Linear"
                f1_str$ = f1_str$+string$(f1)+tab$
                f2_str$ = f2_str$+string$(f2)+tab$
            endfor
            appendFileLine: out$, left$(name$,1)+tab$+right$(name$,2)+tab$+lbl$+tab$+vow$+tab$+f1_str$+f2_str$+tab$+string$(incl)
        endif
    endfor
    
endfor