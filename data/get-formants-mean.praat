path$ = "D:\tmp\"
out$ = path$+"formants-mean.txt"

writeFileLine: out$, "speaker"+tab$+"recording"+tab$+"vowel"+tab$+"label"+tab$+"F1"+tab$+"F2"+tab$+"include"

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

            #margin = (end - start)/6
            #start = start + margin
            #end = end - margin
            select 'formantID'
            f1 = Get mean: 1, start, end, "Hertz"
            f2 = Get mean: 2, start, end, "Hertz"
            appendFileLine: out$, left$(name$,1)+tab$+right$(name$,2)+tab$+lbl$+tab$+vow$+tab$+string$(f1)+tab$+string$(f2)+tab$+string$(incl)
        endif
    endfor
    
endfor