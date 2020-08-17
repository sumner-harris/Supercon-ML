def ChemSplit(a):
  #To let my string ends with a capital letter:
  a=a+'A'
  s=0
  blist=[]

  for ii in range(1,len(a)):
    if str(a[ii]).isupper():
      
      blist.append(a[s:ii])
      s=ii
      
  #print(blist,'\n')

  clist=[]
  dlist=[]

  for ii in range(len(blist)):
    strtemp = blist[ii]

    #Case 1: only 1 element
    if len(strtemp) == 1:
      clist.append(strtemp)
      dlist.append(1)

    else:

      #Case 2: the second character is "number":
      if (ord(strtemp[1])) >= 48:
        if (ord(strtemp[1])) <= 57:
          clist.append(strtemp[0])
          dlist.append(float(strtemp[1:])) 

      #Case 3: the second character is "little letter":
      if(ord(strtemp[1])) >= 97:
        if(ord(strtemp[1])) <= 122:
          clist.append(strtemp[0:2])
              
          if len(strtemp) == 2:
            dlist.append(1)
          else:
            dlist.append(float(strtemp[2:]))
                
  return clist, dlist


################################################################

#Main Program:
#a='F1.33333333Sn1Pb1.33333333'

#print(a,'\n')
#clist, dlist = ChemSplit(a)

#print(clist)
#print(dlist)

