import numpy as np
from numba import njit,jit
import numba
from numba.typed import List
import sys
@njit
def getActionSize():
    return 77
@njit
def getStateSize():
    return 128
@njit
def getAgentSize():
    return 6
@njit
def initEnv():
    """init env"""
    env = np.zeros(99)
    card = np.arange(65) #card except Defuse and Explo kitten and Implo Kitten
    np.random.shuffle(card)
    env[0:76] = 7 # 7 is card on draw pile
    for i in range(6): # draw 7 card for player: id from 0 to 5
        env[card[i*7:(i+1)*7]] = i
        env[65+i] = i
    draw_pile = np.where(env[0:76]==7.)[0].astype(np.float64)
    np.random.shuffle(draw_pile)
    discard_pile = np.zeros(19)#card on discard pile will have id 6

    env[76] = 0 # nope count
    env[77] = 0 # track player id main turn 
    env[78:83] = [2,3,4,5,0] # track player id Nope turn
    env[83:89] = 1 # 0 if lose else 1
    env[89] = 0 #phase [0:main turn, 1:nope turn,2:steal card turn,3:choose/take card turn]
    env[90] = 1 # number of card player env[77] have to draw
    env[91:94] = [0,0,0] #three card in see the future
    env[94] = -1 # player env[77] last action
    env[95] = env[77]+1
    env[96] = -1 #player chosen in phase 2

    env[97] = 1 # main direction
    env[98] = 0 # state of the Imploding Kitten
    return env,draw_pile,discard_pile

@njit
def getNumCard(env,idx):
    """Get the number of card with given type"""
    return np.where(env==idx)[0].shape[0]
@njit
def getAllNumCard(env,idx):
    """Get all the number of card """
    state = np.zeros(17)
    state[0] = getNumCard(env[0:5],idx)
    for i in range(4):
        state[1+i] = getNumCard(env[5+i*4:9+i*4],idx)
    state[5] = getNumCard(env[21:26],idx)
    for i in range(9):
        state[6+i] = getNumCard(env[26+i*4:30+i*4],idx)
    state[15] = getNumCard(env[62:65],idx)
    state[16] = getNumCard(env[65:71],idx)
    return state
@njit
def getCardType(id):
    cards = List([np.arange(0.,5.),np.arange(5.,9.),np.arange(9.,13.),np.arange(13.,17.),np.arange(17.,21.),np.arange(21.,26.),np.arange(26.,30.),np.arange(30.,34.),np.arange(34.,38.),np.arange(38.,42.),np.arange(42.,46.),np.arange(46.,50.),np.arange(50.,54.),np.arange(54.,58.),np.arange(58.,62.),np.arange(62.,65.),np.arange(65.,71.),np.arange(71.,75.),np.arange(75.,76.)])
    i = 0
    for c in cards:
        if id in c:
            return i
        i+=1
@njit
def visualCard(card):
    arr = []
    lst = ['Nope','Attack','Skip','Favor','Shuffle','See the future','TCT','RRC','BC','HPC','CTM','Reverse','Draw Bottom','Feral Cat','Alter the Future','Targeted Attack','Defuse','Exploding Kitten','Imploding Kitten']
    for i in card:
        if i!=-1:
            arr.append(lst[int(getCardType(i))])
    return arr
@njit
def getCardRange(type_card):
    """Get the range of the card given type"""
    cards = List([np.arange(0.,5.),np.arange(5.,9.),np.arange(9.,13.),np.arange(13.,17.),np.arange(17.,21.),np.arange(21.,26.),np.arange(26.,30.),np.arange(30.,34.),np.arange(34.,38.),np.arange(38.,42.),np.arange(42.,46.),np.arange(46.,50.),np.arange(50.,54.),np.arange(54.,58.),np.arange(58.,62.),np.arange(62.,65.),np.arange(65.,71.),np.arange(71.,75.),np.arange(75.,76.)])
    return cards[type_card].astype(np.int64)[0],cards[type_card].astype(np.int64)[-1]+1

@njit
def getAgentState(env,draw_pile,discard_pile):
    state = np.zeros(getStateSize())
    #get card
    if env[89]==1: #Nope phase
        state[0:17] = getAllNumCard(env,env[95])
        state[17:35] = discard_pile[:18] #discard pile
        state[35] = np.where(draw_pile!=-1)[0].shape[0] #number of card in draw pile
        state[36] = np.where(env[83:89]==1)[0].shape[0]
        state[95:100][int(env[89])] = 1 #phase
        if env[94]>=0:
            state[101:116][int(env[94])] = 1# player main t
        nope_turn = nopeTurn(env[95],reverse=(env[97]==0))
        for i in range(5):
            state[116+i] = env[83:89][int(nope_turn[i])]
        state[121] = env[83:89][int(env[95])] #lose or not

    elif env[89]==3 and env[94]==3: #
        state[0:17] = getAllNumCard(env,env[96])
        state[17:35] = discard_pile[:18] #discard pile
        state[35] = np.where(draw_pile!=-1)[0].shape[0] #number of card in draw pile
        state[36] = np.where(env[83:89]==1)[0].shape[0]
        state[95:100][int(env[89])] = 1 #phase
        if env[94]>=0:
            state[101:116][int(env[94])] = 1# player main turn last action
        nope_turn = nopeTurn(env[96],reverse=(env[97]==0))
        for i in range(5):
            state[116+i] = env[83:89][int(nope_turn[i])]
        state[121] = env[83:89][int(env[96])] #lose or not
    else:
        state[0:17] = getAllNumCard(env,env[77])
        state[17:35] = discard_pile[:18] #discard pile
        state[35] = np.where(draw_pile!=-1)[0].shape[0] #number of card in draw pile
        state[36] = np.where(env[83:89]==1)[0].shape[0]
        state[37] = env[76]%2 #1 if action been Nope else 0
        for i in range(3):
            if env[91+i]!=-1:
                card = np.zeros(19)
                card[int(getCardType(env[91+i]))] = 1
                state[37+19*i:56+19*i] = card# three card if use see the future
        state[95:100][int(env[89])] = 1 #phase
        state[100] = env[90] # number of card player have to draw
        if env[94]>=0:
            state[101:116][int(env[94])] = 1# player main turn last action
        for i in range(5):
            state[116+i] = env[83:89][int(env[78+i])]
        state[121] = env[83:89][int(env[77])] #lose or not
    for i in range(5):
        state[122+i] = np.where(env[0:76]==env[78+i])[0].shape[0]
    state[127] = draw_pile[18]
    return state
@njit
def getValidActions(state):
    list_action = np.zeros(getActionSize())
    if state[95]==1:#main turn
        list_action[1:6] = (state[1:6]>0).astype(np.float64)
        if np.sum(state[122:127])==0:
            list_action[3] = 0
        list_action[6] = 1
        if np.max(state[0:16])>=2 and np.sum(state[122:127])>0:#two of a kind
            list_action[7] = 1
        elif np.max(state[6:11]) + state[13]>=2 and np.sum(state[122:127])>0:
            list_action[7] = 1

        
        if np.max(state[0:16])>=3 and np.sum(state[122:127])>0:#three of a kind
            list_action[8] = 1
        elif np.max(state[6:11]) + state[13]>=3 and np.sum(state[122:127])>0:
            list_action[8] = 1
        type_card = (state[0:16]>0).astype(np.float64)
        if (np.sum(type_card)>=5 or np.sum((state[6:11]>0).astype(np.float64))+state[13]>=5)  and np.sum(state[17:34])>0:#five of a kind
            list_action[9] = True
        list_action[11:13] = (state[11:13]>0).astype(np.float64) #4 new action
        list_action[13:15] = (state[14:16]>0).astype(np.float64) 
    elif state[96]==1: #Nope turn
        if state[0]>0:
            list_action[0] = 1 #Nope
        list_action[10] = 1 #skip Nope

    elif state[97]==1: #Choose player
        for i in range(5):
            if state[116+i]==1 and state[122+i]>0:
                list_action[15+i] = 1
        if np.sum(list_action[15:20])==0:
            list_action[6] = 1

        
    elif state[98]==1: #choose/take card turn
        main_action =  np.where(state[101:116]==1)[0][0]
        if main_action==3:
            list_action[20:37][np.where(state[0:17]>0)] = 1
        elif main_action==8:
            list_action[37:54] = 1
        elif main_action==9:
            list_action[54:71] = (state[17:34]>0) * 1.0
    elif state[99]==1: #Alter future phase
        ##list_change = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])
        if state[35]>=3:
            list_action[71:77] = 1
        elif state[35]==2:
            list_action[71] = 1
            list_action[73] = 1
        else:
            list_action[71] = 1
    if np.sum(list_action)==0:
        list_action[10] = 1

    return list_action
@njit
def checkDefuse(env,discard_pile): # get the Defuse (if player have else -1)

    card = np.where(env[65:71]==env[77])[0].astype(np.int64)
    if card.shape[0] > 0:
        card_id = card[0]
        env[65:71][card_id] = 6
        discard_pile[16]+=1
        #print('Player ',env[77],' have Defuse!')
        return True
    return False
@njit
def checkExploding(card): # check if that card is expode or not
    explode = np.array([71.,72.,73.,74.],dtype=np.float64)
    if card in explode:
        #print('Player draw an Exploding kitten!')
        return True
    return False

@njit
def checkImploding(card):
    #return true if player draw an Imploding Kitten.
    if card==75:
        #print('Player draw an Imploding Kitten!')
        return True
    return False
@njit
def nopeTurn(idx,reverse=False):
        if reverse==False:
                return np.arange(idx+1.,idx+6.) % 6
        else:
                return np.arange(idx+5.,idx,-1) % 6
@njit
def changeTurn(env,num_card_draw=1,reverse=False):
    """Change the main turn"""
    if reverse==False:
        env[77] = int(env[77]+1)%6
        while env[83:89][int(env[77])]==0:#if player id is already lost.
            env[77] = int(env[77]+1)%6
        env[78:83] = nopeTurn(env[77],env[97]==0)
        env[76] = 0 #reset nope count
    else:
        env[77] = int(env[77]-1)%6
        while env[83:89][int(env[77])]==0:#if player id is already lost.
            env[77] = int(env[77]-1)%6
        env[78:83] = nopeTurn(env[77],env[97]==0)
        env[76] = 0 #reset nope count
    for i in range(5):
        if env[83:89][int(env[78:83][i])]==1:
            env[95] = env[78:83][i] #reset nope player id
    # if env[90]:
    if env[90]>=2:
        env[90] += num_card_draw #card next player draw
    else:
        env[90] = num_card_draw
    env[89] = 0 # change phase to 0
    env[91:94] = 0
    return env
@njit
def drawCard(env,draw_pile,discard_pile,from_bottom=False,change_turn=True):
    """Draw card"""
    #print('Player ',env[77],' draw ', env[90],'card(s)')
    if from_bottom==True:
        num_cards = 1
    else:
        num_cards = env[90]
    for i in range(int(num_cards)):
        if from_bottom==True:
            index_draw = np.where(draw_pile!=-1)[0][-1]
        else:
            index_draw = np.where(draw_pile!=-1)[0][0]
        #print(f'Draw : {visualCard([draw_pile[index_draw]])[0]}')
        if checkExploding(draw_pile[index_draw]):#draw an exploding kitten
            if checkDefuse(env,discard_pile):#player have defuse
                idx = np.random.randint(index_draw,draw_pile.shape[0])
                draw_pile_2 = np.zeros_like(draw_pile)
                draw_pile_2[0:index_draw] = draw_pile[0:index_draw]
                draw_pile_2[index_draw:idx] = draw_pile[index_draw+1:idx+1]
                draw_pile_2[idx] = draw_pile[index_draw]
                draw_pile_2[idx+1:] = draw_pile[idx+1:]
                draw_pile = draw_pile_2
                #insert explode card back to the Draw Pile
            else:#player lost
                #print('Player ',env[77],' loss!')
                env[83:89][int(env[77])] = 0
                env[0:76][np.where(env[0:76]==env[77])] = 6
                env[71:75][np.where(env[71:75]!=6)] = 6
                discard_pile[17]+=1
                draw_pile[index_draw] = -1
                change_turn = True
                break
        elif checkImploding(draw_pile[index_draw]):
            if env[98]==0:#face down
                env[98] = 1
                idx = np.random.randint(index_draw,draw_pile.shape[0])
                draw_pile_2 = np.zeros_like(draw_pile)
                draw_pile_2[0:index_draw] = draw_pile[0:index_draw]
                draw_pile_2[index_draw:idx] = draw_pile[index_draw+1:idx+1]
                draw_pile_2[idx] = draw_pile[index_draw]
                draw_pile_2[idx+1:] = draw_pile[idx+1:]
                draw_pile = draw_pile_2
            else: #face up
                #print('Player ',env[77],' loss!')
                env[83:89][int(env[77])] = 0
                env[0:76][np.where(env[0:76]==env[77])] = 6
                env[75] = 6
                discard_pile[18]+=1
                draw_pile[index_draw] = -1
                change_turn = True
                break
            
        else:#draw other card
            env[0:76][int(draw_pile[index_draw])] = env[77] #draw
            draw_pile[index_draw] = -1
    
    if change_turn==True:
        env[90] = 0
        env = changeTurn(env,1,env[97]==0)
    return env,draw_pile,discard_pile
@njit
def checkIfNope(env):
    """Return True if the main player's card has been Nope"""
    return env[76]%2==1

@njit
def executeMainAction(env,draw_pile,discard_pile,action):
    """Execute main action if it has not been Nope"""
    #print('Execute main Action!')
    env[76] = 0
    if action==1: #Attack
        #print(f'Player {env[77]} attack!')
        env[89] = 0
        env = changeTurn(env,num_card_draw=2) #change main turn, next player draw 2 card
    elif action==2: #Skip
        #print(f'Player {env[77]} skip!')
        env[90]-=1
        env[89] = 0
        if env[90]==0:
            env = changeTurn(env,num_card_draw=1)
    elif action==3:
        #print(f'Player {env[77]} use favor!')
        env[89] = 2    
    elif action==4: #Shuffle
        #print(f'Player {env[77]} shuffle!')
        np.random.shuffle(draw_pile)
        env[89] = 0
    elif action==5: #See the future
        #print(f'Player {env[77]} see the future!')
        if np.where(draw_pile!=-1)[0].shape[0]>=3:
            env[91:94] = draw_pile[np.where(draw_pile!=-1)[0][0:3]]
        else:
            env[91:94] = np.concatenate((draw_pile[np.where(draw_pile!=-1)[0][0:3]],np.zeros(3-np.where(draw_pile!=-1)[0].shape[0])-1))
        env[89] = 0
    elif action==7:
        #print(f'Player {env[77]} use two of a kind!')
        env[89] = 2
    elif action==8:
        #print(f'Player {env[77]} use three of a kind!')
        env[89] = 2
    elif action==9:
        #print(f'Player {env[77]} use five different cards!')
        env[89] = 3
    elif action==11: #Reverse
        env[97] = (env[97]+1)%2 #change the direction
        env[90]-=1
        if env[90]==0:
            changeTurn(env,1,env[97]==0)
        env[89] = 0
        #print(f'Player {env[77]} use Reverse!')
    elif action==12: #Draw from bottom
        #print(f'Player {env[77]} use Draw from Bottom!')
        if env[90]>1:
            env,draw_pile,discard_pile = drawCard(env,draw_pile,discard_pile,from_bottom=True,change_turn=False)
            env[90]-=1
            env[89] = 0
        else:
            env,draw_pile,discard_pile = drawCard(env,draw_pile,discard_pile,from_bottom=True,change_turn=True)
    elif action==13: #Alter the future
        #print(f'Player {env[77]} use Alter the future!')
        if np.where(draw_pile!=-1)[0].shape[0]>=3:
            env[91:94] = draw_pile[np.where(draw_pile!=-1)[0][0:3]]
        else:
            env[91:94] = np.concatenate((draw_pile[np.where(draw_pile!=-1)[0][0:3]],np.zeros(3-np.where(draw_pile!=-1)[0].shape[0])-1))
        env[89] = 4 #special phase of alter the future
    elif action==14: # Targeted Attack
        #print(f'Player {env[77]} use Targeted Attack!')
        env[89] = 2 #choose player to attack
        

    
    return env,draw_pile,discard_pile

@njit
def discardCardNormalAction(env,last_action,discard_pile):
    if last_action==0:
        discard_pile[0]+=1
        env[0:5][np.where(env[0:5]==env[77])[0][0]] = 6
    elif last_action==1: #Attack
        env[5:9][np.where(env[5:9]==env[77])[0][0]] = 6
        discard_pile[1]+=1
    elif last_action==2: #Skip
        env[9:13][np.where(env[9:13]==env[77])[0][0]] = 6
        discard_pile[2]+=1
    elif last_action==3:
        discard_pile[3]+=1
        env[13:17][np.where(env[13:17]==env[77])[0][0]] = 6     
    elif last_action==4: #Shuffle
        env[17:21][np.where(env[17:21]==env[77])[0][0]] = 6
        discard_pile[4]+=1
    elif last_action==5: #See the future
        env[21:26][np.where(env[21:26]==env[77])[0][0]] = 6
        discard_pile[5]+=1
    elif last_action==11: #Reverse
        env[46:50][np.where(env[46:50]==env[77])[0][0]] = 6
        discard_pile[11]+=1
    elif last_action==12:#Draw from Bottom
        env[50:54][np.where(env[50:54]==env[77])[0][0]] = 6
        discard_pile[12]+=1
    elif last_action==13: #Alter the future
        env[58:62][np.where(env[58:62]==env[77])[0][0]] = 6
        discard_pile[14]+=1
    elif last_action==14: #Targeted Attack
        env[62:65][np.where(env[62:65]==env[77])[0][0]] = 6
        discard_pile[15]+=1

@njit
def discardCardSpecialAction(env,last_action,discard_pile):
    """Discard card after using special action"""
    all_num_card = getAllNumCard(env,env[77])
    num_cat = all_num_card[6:11]
    num_special = all_num_card[0:6]
    if last_action==7: # two of a kind
        if np.max(num_cat)>=2:
            if 2. in num_cat:
                type_card = np.where(num_cat==2)[0][0]+6
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])] = 6
                discard_pile[int(type_card)]+=2

            else:
                type_card = np.random.choice(np.where(num_cat>=2)[0])+6
                for i in range(2):
                    env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6
                    discard_pile[int(type_card)]+=1
        elif np.max(num_cat)+all_num_card[13] >=2: #Feral cat
            max_cat = np.max(num_cat)
            if max_cat == 1:
                type_card = np.where(num_cat==1)[0][0]+6
                env[54:58][np.where(env[54:58]==env[77])[0][0]] = 6
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6
            else:
                for i in range(2):
                    env[54:58][np.where(env[54:58]==env[77])[0][0]] = 6
        else:
            type_card = np.random.choice(np.where(all_num_card>=2)[0])
            for i in range(2):
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6
                discard_pile[int(type_card)]+=1
            
    elif last_action==8:
        if np.max(num_cat)>=3:
            if 3 in num_cat:
                type_card = np.where(num_cat==3)[0][0]+6
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])] = 6
                discard_pile[int(type_card)]+=3
            else:
                type_card = np.random.choice(np.where(num_cat>=3)[0])+6
                for i in range(3):
                    env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6
                    discard_pile[int(type_card)]+=1
        elif np.max(num_cat)+all_num_card[13] >=3: #Feral cat
            max_cat = np.max(num_cat)
            if max_cat == 1:
                type_card = np.where(num_cat==1)[0][0]+6
                for i in range(2):
                    env[54:58][np.where(env[54:58]==env[77])[0][0]] = 6
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6
            elif max_cat==2:
                type_card = np.where(num_cat==2)[0][0]+6
                env[54:58][np.where(env[54:58]==env[77])[0][0]] = 6
                for i in range(2):
                    env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6
            else:
                for i in range(3):
                    env[54:58][np.where(env[54:58]==env[77])[0][0]] = 6
        else:
            type_card = np.random.choice(np.where(all_num_card>=3)[0])
            for i in range(3):
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6
                discard_pile[int(type_card)]+=1
    elif last_action==9:
        if np.sum((num_cat>0).astype(np.float64))==5:
            for i in range(5):
                type_card = 6+i
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6
                discard_pile[int(type_card)]+=1
        elif np.sum((num_cat>0).astype(np.float64))+all_num_card[13] >= 5:
            num_type_cat = np.sum((num_cat>0).astype(np.float64))
            for i in range(int(5-num_type_cat)):
                env[54:58][np.where(env[54:58]==env[77])[0][0]] = 6
            for i in range(5):
                type_card = 6+i
                if all_num_card[type_card] > 0:
                    env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6

        else:

            num_type_cat = np.sum((num_cat>0).astype(np.float64))
            for i in range(int(all_num_card[13])):
                env[54:58][np.where(env[54:58]==env[77])[0][0]] = 6
            for i in range(5):
                type_card = 6+i
                if all_num_card[type_card] > 0:
                    env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6
            num_spec = 5 - num_type_cat - all_num_card[13]
            special_card = np.concatenate((np.where(all_num_card[0:6]>0)[0],np.where(all_num_card[11:13]>0)[0]+11,np.where(all_num_card[14:16]>0)[0]+14))
            np.random.shuffle(special_card)
            for i in range(int(num_spec)):
                type_card = special_card[0]
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[77])[0][0]] = 6
                special_card = special_card[1:]
                discard_pile[int(type_card)]+=1

    return env,discard_pile
@njit
def idPlayerCanUseNope(env,nope_id,reverse=False):
    """return the id of the player that have the nope card, else -1"""
    main_id = env[77]
    nope_turn = nopeTurn(main_id,reverse=(env[97]==0))

    idx_old = -1
    for i in range(5):
        if nope_turn[i] == nope_id:
            idx_old = i
            break
    else:
        idx_old = -1
    if idx_old+1==5:
        return main_id
    else:
        for i in range(idx_old+1,5):
            idx = nope_turn[i]
            if np.where(env[0:5]==idx)[0].shape[0]>=1 and env[83:89][int(idx)] == 1:
                return idx
        return main_id

@njit
def stepEnv(env,draw_pile,discard_pile,action):
    phase = env[89]
    main_id = env[77]
    nope_id = env[95]
    nope_count = env[76]
    last_action = env[94]
    if phase==0: #Phase 0: Main Turn
        env[76] = 0
        if action==6: #draw card
            env,draw_pile,discard_pile = drawCard(env,draw_pile,discard_pile)
        else:
            env[94] = action
            if env[94]<=5 or env[94]>=11:
                discardCardNormalAction(env,env[94],discard_pile)
            elif env[94]>=7 and env[94]<=9:
                discardCardSpecialAction(env,env[94],discard_pile)

            env[95] = idPlayerCanUseNope(env,main_id,env[97]==1)
            if env[95]==main_id:
                env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[94])
                #print(f'Action {env[94]} has been executed!')
            else:
                env[89] = 1 #change to Nope phase
    elif phase==1:#Phase 1: Nope phase

        if action==0 and env[95]!=main_id: #other player use Nope
            #print(f'Player {env[95]} use Nope!')
            env[76]+=1 # increase Nope Count
            env[0:5][np.where(env[0:5]==env[95])[0][0]] = 6
            discard_pile[0]+=1
            env[95] = idPlayerCanUseNope(env,env[95],env[97]==0)
            if env[95]==main_id:
                if not checkIfNope(env): #if not been Nope
                    env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[94])
                    #print(f'Action {env[94]} has been executed!')
        elif action==0 and env[95]==main_id:
                env[76]+=1 # increase Nope Count
                env[0:5][np.where(env[0:5]==env[77])[0][0]] = 6
                if not checkIfNope(env): #if not been Nope
                    env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[94])
                    #print(f'Action {env[94]} has been executed!')
        else:
            if env[95]==main_id:
                if not checkIfNope(env): #if not been Nope
                    env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[94])
                    #print(f'Action {env[94]} has been executed!')
                else: # if Nope
                    if action==0:
                        #print('Main player use Yup!')
                        env[76] = 0 #reset to original
                        env[0:5][np.where(env[0:5]==main_id)[0][0]] = 6
                        discard_pile[0]+=1
                        env[95] = idPlayerCanUseNope(env,env[95],env[97]==0)
                        if env[95]==main_id:
                            env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[94])
                            #print(f'Action {env[94]} has been executed!')
                    else:
                        #print(f'Action {env[94]} has been Nope!')
                        env[94] = -1# action has been Nope
                        env[89] = 0 # back to phase 0
                        env[95] = idPlayerCanUseNope(env,main_id,env[97]==0)
            # env[95] = idPlayerCanUseNope(env,env[95])   
            else:
                env[95] = idPlayerCanUseNope(env,env[95],env[97]==0)
                if env[95]==main_id:
                    if not checkIfNope(env): #if not been Nope
                        env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[94])
                        #print(f'Action {env[94]} has been executed!')
     
    elif phase==2:# phase 2: choose player to steal card. Only main_id can enter this phase
        if action==6:
            env,draw_pile,discard_pile = drawCard(env,draw_pile,discard_pile)
        else:
            env[96] = env[78:83][int(action-15)]
            last_action = env[94]
            #print(f'Player {env[77]} choose player {env[96]} to steal!')
            if last_action==7:
                card_on_player_chosen = np.where(env[0:76]==env[96])[0]
                card = np.random.choice(card_on_player_chosen)
                env[0:76][card] = env[77]
                #used card go to Discard Pile
                env[89] = 0
            elif last_action==14:
                env[77] = env[96]
                env[90] = 2
                env[89] = 0
            else:
                env[89] = 3
        env[76] = 0 

    elif phase==3: #phase 3: choose card to give/take. Only main_id can enter this phase
        last_action = env[94]
        if last_action==3:
            type_card = action - 20
            all_card_to_take = np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[96])[0]
            env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][int(all_card_to_take[0])] = env[77]


        elif last_action==8:
            #take card
            type_card = action - 37
            all_card_to_take = np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[96])[0]
            if all_card_to_take.shape[0]>0:
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][int(all_card_to_take[0])] = env[77]
            #used card go to Discard Pile
        elif last_action==9:
            type_card = action - 54
            if np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==6)[0].shape[0]>0:
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==6)[0][0]] = env[77]
        env[94] = -1
        env[89] = 0
        env[76] = 0

    elif phase==4: #Special phase: Alter the future
        act = int(action - 71)
        list_change = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])
        env[91:94] = env[91:94][list_change[act]]
        index_future = np.where(draw_pile!=-1)[0]
        if index_future.shape[0]>=3:
            draw_pile[index_future[:3]] = draw_pile[index_future[:3]][list_change[act]]
        elif index_future.shape[0]==2:
            if act == 2:
                draw_pile[index_future] = draw_pile[index_future][np.array([1,0])]
        env[94] = -1
        env[89] = 0
        env[76] = 0
    return env,draw_pile,discard_pile

@njit
def checkEnded(env):
    if np.sum(env[83:89])==1:
        return np.where(env[83:89]==1)[0][0]
    else:
        return -1
@njit
def getReward(state):
    if state[121] == 0:
        return 0
    elif np.sum(state[116:121])==0:
        return 1
    else:
        return -1
@njit
def random_player(state,per):
    list_action  = np.where(getValidActions(state)==1)[0]
    action = np.random.choice(list_action)
    #print(list_action)
    return action,per

@njit()
def bot_lv0(state, perData):
    validActions = getValidActions(state)
    arr_action = np.where(validActions==1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData

@njit
def one_game_numba(p0,pIdOrder,per_player,per1,per2,per3,per4,per5,p1,p2,p3,p4,p5):
    env,draw_pile,discard_pile = initEnv()

    winner = -1
    turn = 0
    while True:
        turn +=1
        phase = env[89]
        main_id = env[77]
        nope_id = env[95]
        last_action = env[94]
        if phase==0:
            pIdx = int(main_id)
        elif phase==1:
            pIdx = int(nope_id)
        elif phase==2:
            pIdx = int(main_id)
        elif phase==3:
            if last_action==3:
                pIdx = int(env[96])
            else:
                pIdx = int(main_id)
        elif phase==4:
            pIdx = int(main_id)
        if pIdOrder[pIdx] == -1:
            action, per_player = p0(getAgentState(env,draw_pile,discard_pile), per_player)
        elif pIdOrder[pIdx] == 1:
            action, per1 = p1(getAgentState(env,draw_pile,discard_pile), per1)
        elif pIdOrder[pIdx] == 2:
            action, per2 = p2(getAgentState(env,draw_pile,discard_pile), per2)
        elif pIdOrder[pIdx] == 3:
            action, per3 = p3(getAgentState(env,draw_pile,discard_pile), per3)
        elif pIdOrder[pIdx] == 4:
            action, per4 = p4(getAgentState(env,draw_pile,discard_pile), per4)
        elif pIdOrder[pIdx] == 5:
            action, per5 = p5(getAgentState(env,draw_pile,discard_pile), per5)
        env,draw_pile,discard_pile = stepEnv(env,draw_pile,discard_pile,action)
        
        winner = checkEnded(env)
        if winner != -1 or turn>300:
            break
    for idx in range(6):
        env[77] = idx
        if pIdOrder[pIdx] == -1:
            action, per_player = p0(getAgentState(env,draw_pile,discard_pile), per_player)
        elif pIdOrder[pIdx] == 1:
            action, per1 = p1(getAgentState(env,draw_pile,discard_pile), per1)
        elif pIdOrder[pIdx] == 2:
            action, per2 = p2(getAgentState(env,draw_pile,discard_pile), per2)
        elif pIdOrder[pIdx] == 3:
            action, per3 = p3(getAgentState(env,draw_pile,discard_pile), per3)
        elif pIdOrder[pIdx] == 4:
            action, per4 = p4(getAgentState(env,draw_pile,discard_pile), per4)
        elif pIdOrder[pIdx] == 5:
            action, per5 = p5(getAgentState(env,draw_pile,discard_pile), per5)

    win = False        
    if np.where(pIdOrder == -1)[0][0] == checkEnded(env): 
        win = True
    else: 
        win = False

    return win, per_player

@njit()
def n_game_numba(p0, num_game, per_player, list_other, per1, per2, per3, per4, per5, p1, p2, p3, p4, p5):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_numba(p0, list_other, per_player, per1, per2, per3, per4, per5, p1, p2, p3, p4, p5)
        win += winner
    return win, per_player

import importlib.util, json, sys
from setup import SHORT_PATH

def load_module_player(player):
    return  importlib.util.spec_from_file_location('Agent_player', f"{SHORT_PATH}Agent/{player}/Agent_player.py").loader.load_module()

@njit()
def random_Env(p_state, per):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], per

def numba_main_2(p0, num_game, per_player, level, *args):
    num_bot = getAgentSize() - 1
    list_other = np.array([-1] + [i+1 for i in range(num_bot)])
    try: check_njit = check_run_under_njit(p0, per_player)
    except: check_njit = False

    if "_level_" not in globals():
        global _level_
        _level_ = level
        init = True
    else:
        if _level_ != level:
            _level_ = level
            init = True
        else:
            init = False

    if init:
        global _list_per_level_
        global _list_bot_level_
        _list_per_level_ = []
        _list_bot_level_ = []

        if _level_ == 0:
            _list_per_level_ = [np.array([[0.]], dtype=np.float64) for _ in range(num_bot)]
            _list_bot_level_ = [bot_lv0 for _ in range(num_bot)]
        else:
            env_name = sys.argv[1]
            if len(args) > 0:
                dict_level = json.load(open(f'{SHORT_PATH}Log/check_system_about_level.json'))
            else:
                dict_level = json.load(open(f'{SHORT_PATH}Log/level_game.json'))

            if str(_level_) not in dict_level[env_name]:
                raise Exception('Hiện tại không có level này')

            lst_agent_level = dict_level[env_name][str(level)][2]
            lst_module_level = [load_module_player(lst_agent_level[i]) for i in range(num_bot)]
            for i in range(num_bot):
                data_agent_level = np.load(f'{SHORT_PATH}Agent/{lst_agent_level[i]}/Data/{env_name}_{level}/Train.npy',allow_pickle=True)
                _list_per_level_.append(lst_module_level[i].convert_to_test(data_agent_level))
                _list_bot_level_.append(lst_module_level[i].Test)

    if check_njit:
        return n_game_numba(p0, num_game, per_player, list_other,
                                _list_per_level_[0], _list_per_level_[1], _list_per_level_[2],_list_per_level_[3],_list_per_level_[4],
                                _list_bot_level_[0], _list_bot_level_[1], _list_bot_level_[2],_list_bot_level_[3],_list_bot_level_[4])
    else:
        return n_game_normal(p0, num_game, per_player, list_other,
                                _list_per_level_[0], _list_per_level_[1], _list_per_level_[2],_list_per_level_[3],_list_per_level_[4],
                                _list_bot_level_[0], _list_bot_level_[1], _list_bot_level_[2],_list_bot_level_[3],_list_bot_level_[4])
    
def one_game_normal(p0,pIdOrder,per_player,per1,per2,per3,per4,per5,p1,p2,p3,p4,p5):

    env,draw_pile,discard_pile = initEnv()

    winner = -1
    turn = 0
    while True:
        turn +=1
        phase = env[89]
        main_id = env[77]
        nope_id = env[95]
        last_action = env[94]
        if phase==0:
            pIdx = int(main_id)
        elif phase==1:
            pIdx = int(nope_id)
        elif phase==2:
            pIdx = int(main_id)
        elif phase==3:
            if last_action==3:
                pIdx = int(env[96])
            else:
                pIdx = int(main_id)
        elif phase==4:
            pIdx = int(main_id)
        if pIdOrder[pIdx] == -1:
            action, per_player = p0(getAgentState(env,draw_pile,discard_pile), per_player)
        elif pIdOrder[pIdx] == 1:
            action, per1 = p1(getAgentState(env,draw_pile,discard_pile), per1)
        elif pIdOrder[pIdx] == 2:
            action, per2 = p2(getAgentState(env,draw_pile,discard_pile), per2)
        elif pIdOrder[pIdx] == 3:
            action, per3 = p3(getAgentState(env,draw_pile,discard_pile), per3)
        elif pIdOrder[pIdx] == 4:
            action, per4 = p4(getAgentState(env,draw_pile,discard_pile), per4)
        elif pIdOrder[pIdx] == 5:
            action, per5 = p5(getAgentState(env,draw_pile,discard_pile), per5)
        env,draw_pile,discard_pile = stepEnv(env,draw_pile,discard_pile,action)
        
        winner = checkEnded(env)
        if winner != -1 or turn>300:
            break
    for idx in range(6):
        env[77] = idx
        if pIdOrder[pIdx] == -1:
            action, per_player = p0(getAgentState(env,draw_pile,discard_pile), per_player)
        elif pIdOrder[pIdx] == 1:
            action, per1 = p1(getAgentState(env,draw_pile,discard_pile), per1)
        elif pIdOrder[pIdx] == 2:
            action, per2 = p2(getAgentState(env,draw_pile,discard_pile), per2)
        elif pIdOrder[pIdx] == 3:
            action, per3 = p3(getAgentState(env,draw_pile,discard_pile), per3)
        elif pIdOrder[pIdx] == 4:
            action, per4 = p4(getAgentState(env,draw_pile,discard_pile), per4)
        elif pIdOrder[pIdx] == 5:
            action, per5 = p5(getAgentState(env,draw_pile,discard_pile), per5)

    win = False        
    if np.where(pIdOrder == -1)[0][0] == checkEnded(env): 
        win = True
    else: 
        win = False

    return win, per_player

def n_game_normal(p0, num_game, per_player, list_other, per1, per2, per3, per4, per5, p1, p2, p3, p4, p5):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner,per_player  = one_game_normal(p0, list_other, per_player, per1, per2, per3, per4, per5, p1, p2, p3, p4, p5)
        win += winner
    return win, per_player


@njit()
def check_run_under_njit(Agent):
    return True

