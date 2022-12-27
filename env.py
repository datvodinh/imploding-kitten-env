import numpy as np
from numba import njit
import numba
from numba.typed import List
import sys
@njit
def getActionSize():
    return 51
@njit
def getStateSize():
    return 91
@njit
def initEnv():
    """init env"""
    env = np.zeros(75)
    card = np.arange(46) #card except Defuse and Explo kitten
    np.random.shuffle(card)
    env[:56] += 5 # 5 is card on draw pile
    for i in range(5): # draw 4 card for player: id from 0 to 4
        env[card[i*4:(i+1)*4]] = i
        env[46+i] = i

    draw_pile = np.where(env==5.)[0].astype(np.float64)
    np.random.shuffle(draw_pile)

    discard_pile = np.zeros(13)#card on discard pile will have id 6

    env[56] = 0 # nope count
    env[57] = 0 # track player id main turn 
    env[58:62] = [2,3,4,0] # track player id Nope turn
    env[62:67] = 1 # 0 if lose else 1
    env[67] = 0 #phase [0:main turn, 1:nope turn,2:steal card turn,3:choose/take card turn]
    env[68] = 1 # number of card player env[57] have to draw
    env[69:72] = [0,0,0] #three card in see the future
    env[72] = -1 # player env[57] last action
    env[73] = env[57]+1
    env[74] = -1 #player chosen in phase 2

    return env,draw_pile,discard_pile

@njit
def getNumCard(env,idx):
    """Get the number of card with given type"""
    return np.where(env==idx)[0].shape[0]

@njit
def getAllNumCard(env,idx):
    """Get all the number of card """
    state = np.zeros(12)
    state[0] = getNumCard(env[0:5],idx)
    for i in range(4):
        state[1+i] = getNumCard(env[5+i*4:9+i*4],idx)
    state[5] = getNumCard(env[21:26],idx)
    for i in range(5):
        state[6+i] = getNumCard(env[26+i*4:30+i*4],idx)
    state[11] = getNumCard(env[46:52],idx)
    return state

@njit
def getCardType(idx):
    """Get the type of the card"""
    cards = List([np.arange(0.,5.),np.arange(5.,9.),np.arange(9.,13.),np.arange(13.,17.),np.arange(17.,21.),np.arange(21.,26.),np.arange(26.,30.),np.arange(30.,34.),np.arange(34.,38.),np.arange(38.,42.),np.arange(42.,46.),np.arange(46.,52.),np.arange(52.,56.)])
    i = 0
    for c in cards:
        if idx in c:
            return i
        else:
            i+=1
    
@njit
def getCardRange(type_card):
    """Get the range of the card given type"""
    cards = List([np.arange(0.,5.),np.arange(5.,9.),np.arange(9.,13.),np.arange(13.,17.),np.arange(17.,21.),np.arange(21.,26.),np.arange(26.,30.),np.arange(30.,34.),np.arange(34.,38.),np.arange(38.,42.),np.arange(42.,46.),np.arange(46.,52.),np.arange(52.,56.)])
    return cards[type_card].astype(np.int64)[0],cards[type_card].astype(np.int64)[-1]+1

@njit
def getAgentState(env,draw_pile,discard_pile):
    state = np.zeros(getStateSize())
    #get card
    if env[67]==1:
        state[0:12] = getAllNumCard(env,env[73])
        state[12:25] = discard_pile #discard pile
        state[25] = np.where(draw_pile!=-1)[0].shape[0] #number of card in draw pile
        state[26] = np.where(env[62:67]==1)[0].shape[0]
        state[67:71][int(env[67])] = 1 #phase
        if env[72]>=0:
            state[72:82][int(env[72])] = 1# player main t
        state[86] = env[62:67][int(env[73])] #lose or not

    elif env[67]==3 and env[72]==3: #
        state[0:12] = getAllNumCard(env,env[74])
        state[12:25] = discard_pile #discard pile
        state[25] = np.where(draw_pile!=-1)[0].shape[0] #number of card in draw pile
        state[26] = np.where(env[62:67]==1)[0].shape[0]
        state[67:71][int(env[67])] = 1 #phase
        if env[72]>=0:
            state[72:82][int(env[72])] = 1# player main turn last action
        state[86] = env[62:67][int(env[74])] #lose or not
    else:
        state[0:12] = getAllNumCard(env,env[57])
        state[12:25] = discard_pile #discard pile
        state[25] = np.where(draw_pile!=-1)[0].shape[0] #number of card in draw pile
        state[26] = np.where(env[62:67]==1)[0].shape[0]
        state[27] = env[56]%2 #1 if action been Nope else 0
        for i in range(3):
            if env[69+i]!=-1:
                card = np.zeros(13)
                card[int(getCardType(env[69+i]))] = 1
                state[28+13*i:41+13*i] = card# three card if use see the future
        state[67:71][int(env[67])] = 1 #phase
        state[71] = env[68] # number of card player have to draw
        if env[72]>=0:
            state[72:82][int(env[72])] = 1# player main turn last action
        for i in range(4):
            state[82+i] = env[62:67][int(env[58+i])]
        state[86] = env[62:67][int(env[57])] #lose or not
        for i in range(4):
            state[87+i] = np.where(env[0:56]==env[58+i])[0].shape[0]
    return state

@njit
def getValidActions(state):
    list_action = np.zeros(getActionSize())
    if state[67]==1:#main turn
        list_action[1:6] = (state[1:6]>0).astype(np.float64)
        if np.sum(state[87:91])==0:
            list_action[3] = 0
        list_action[6] = 1
        if np.max(state[0:11])>=2 and np.sum(state[87:91])>0:#two of a kind
            list_action[7] = True
        if np.max(state[0:11])>=3 and np.sum(state[87:91])>0:#three of a kind
            list_action[8] = True
        type_card = (state[0:11]>0).astype(np.float64)
        if np.sum(type_card)>=5:#five of a kind
            list_action[9] = True

    elif state[68]==1: #Nope turn
        if state[0]>0:
            list_action[0] = 1 #Nope
        list_action[10] = 1 #skip Nope

    elif state[69]==1: #steal turn
        for i in range(4):
            if state[82+i]==1 and state[87+i]>0:
                list_action[11+i] = 1
        if np.sum(list_action[11:15])==0:
            list_action[6] = 1


        
    elif state[70]==1: #choose/take card turn
        main_action =  np.where(state[72:82]==1)[0][0]
        if main_action==3:
            list_action[15:27][np.where(state[0:12]>0)] = 1
        elif main_action==8:
            list_action[27:39] = 1
        elif main_action==9:
            list_action[39:51] = 1

    return list_action
    
    
@njit
def checkDefuse(env,discard_pile): # get the Defuse (if player have else -1)

    card = np.where(env[46:52]==env[57])[0].astype(np.int64)
    if card.shape[0] > 0:
        card_id = card[0]
        env[46:52][card_id] = 6
        discard_pile[11]+=1
        #print('Player ',env[57],' have Defuse!')
        return True
    return False

@njit
def checkExploding(card): # check if that card is expode or not
    explode = np.array([52,53,54,55],dtype=np.float64)
    if card in explode:
        #print('PLayer draw an Exploding kitten!')
        return True
    return False
@njit
def changeTurn(env,num_card_draw=1):
    """Change the main turn"""
    env[57] = int(env[57]+1)%5
    while env[62:67][int(env[57])]==0:#if player id is already lost.
        env[57] = int(env[57]+1)%5
    env[58:62] = nopeTurn(env[57])
    env[56] = 0 #reset nope count
    for i in range(4):
        if env[62:67][int(env[58:62][i])]==1:
            env[73] = env[58:62][i] #reset nope player id
    # if env[68]:
    if env[68]>=2:
        env[68] += num_card_draw #card next player draw
    else:
        env[68] = num_card_draw
    env[67] = 0 # change phase to 0
    env[69:72] = 0
    return env

@njit
def drawCard(env,draw_pile,discard_pile):
    """Draw card"""
    #print('Player ',env[57],' draw ', env[68],'card(s)')
    for i in range(int(env[68])):
        index_draw = np.where(draw_pile!=-1)[0][0]
        #print(f'Draw : {draw_pile[index_draw]}')
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
                #print('Player ',env[57],' loss!')
                env[62:67][int(env[57])] = 0
                env[0:56][np.where(env[0:56]==env[57])] = 6
                env[52:56][np.where(env[52:56]!=6)] = 6
                discard_pile[12]+=1
                draw_pile[index_draw] = -1
                break
        else:#draw other card
            env[0:56][int(draw_pile[index_draw])] = env[57] #draw
            draw_pile[index_draw] = -1
    env[68] = 0
    env = changeTurn(env,1)
    return env,draw_pile,discard_pile

@njit
def nopeTurn(id):
    """Nope turn given the main player id"""
    return np.arange(id+1.,id+5.) % 5
    

@njit
def checkIfNope(env):
    """Return True if the main player's card has been Nope"""
    return env[56]%2==1

@njit
def executeMainAction(env,draw_pile,discard_pile,action):
    """Execute main action if it has not been Nope"""
    #print('Execute main Action!')
    env[56] = 0
    if action==1: #Attack
        #print(f'Player {env[57]} attack!')
        env[67] = 0
        env = changeTurn(env,num_card_draw=2) #change main turn, next player draw 2 card
    elif action==2: #Skip
        #print(f'Player {env[57]} skip!')
        env[68]-=1
        env[67] = 0
        if env[68]==0:
            env = changeTurn(env,num_card_draw=1)
    elif action==3:
        #print(f'Player {env[57]} use favor!')
        env[67] = 2    
    elif action==4: #Shuffle
        #print(f'Player {env[57]} shuffle!')
        np.random.shuffle(draw_pile)
        env[67] = 0
    elif action==5: #See the future
        #print(f'Player {env[57]} see the future!')
        if np.where(draw_pile!=-1)[0].shape[0]>=3:
            env[69:72] = draw_pile[np.where(draw_pile!=-1)[0][0:3]]
        else:
            env[69:72] = np.concatenate((draw_pile[np.where(draw_pile!=-1)[0][0:3]],np.zeros(3-np.where(draw_pile!=-1)[0].shape[0])-1))
        env[67] = 0
    elif action==7:
        #print(f'Player {env[57]} use two of a kind!')
        env[67] = 2
    elif action==8:
        #print(f'Player {env[57]} use three of a kind!')
        env[67] = 2
    elif action==9:
        #print(f'Player {env[57]} use five different cards!')
        env[67] = 3
    
    return env,draw_pile,discard_pile

@njit
def discardCardNormalAction(env,last_action,discard_pile):
    if last_action==0:
        discard_pile[0]+=1
        env[0:5][np.where(env[0:5]==env[57])[0][0]] = 6
    elif last_action==1: #Attack
        env[5:9][np.where(env[5:9]==env[57])[0][0]] = 6
        discard_pile[1]+=1
    elif last_action==2: #Skip
        env[9:13][np.where(env[9:13]==env[57])[0][0]] = 6
        discard_pile[2]+=1
    elif last_action==3:
        discard_pile[3]+=1
        env[13:17][np.where(env[13:17]==env[57])[0][0]] = 6     
    elif last_action==4: #Shuffle
        env[17:21][np.where(env[17:21]==env[57])[0][0]] = 6
        discard_pile[4]+=1
    elif last_action==5: #See the future
        env[21:26][np.where(env[21:26]==env[57])[0][0]] = 6
        discard_pile[5]+=1

@njit
def discardCardSpecialAction(env,last_action,discard_pile):
    """Discard card after using special action"""
    all_num_card = getAllNumCard(env,env[57])[:11]
    if last_action==7: # two of a kind
        if np.max(all_num_card[6:11])>=2:
            if 2. in all_num_card[6:11]:
                type_card = np.where(all_num_card[6:11]==2)[0][0]+6
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[57])] = 6
                discard_pile[int(type_card)]+=2
            else:
                type_card = np.random.choice(np.where(all_num_card[6:11]>=2)[0])+6
                for i in range(2):
                    env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[57])[0][0]] = 6
                    discard_pile[int(type_card)]+=1
        else:
            type_card = np.random.choice(np.where(all_num_card[0:6]>=2)[0])
            for i in range(2):
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[57])[0][0]] = 6
                discard_pile[int(type_card)]+=1
            
    elif last_action==8:
        if np.max(all_num_card[6:11])>=3:
            if 3 in all_num_card[6:11]:
                type_card = np.where(all_num_card[6:11]==3)[0][0]+6
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[57])] = 6
                discard_pile[int(type_card)]+=3
            else:
                type_card = np.random.choice(np.where(all_num_card[6:11]>=3)[0])+6
                for i in range(3):
                    env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[57])[0][0]] = 6
                    discard_pile[int(type_card)]+=1
        else:
            type_card = np.random.choice(np.where(all_num_card[0:6]>=3)[0])
            for i in range(3):
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[57])[0][0]] = 6
                discard_pile[int(type_card)]+=1
    elif last_action==9:
        if np.sum((all_num_card[6:11]>0).astype(np.float64))==5:
            for i in range(5):
                type_card = 6+i
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[57])[0][0]] = 6
                discard_pile[int(type_card)]+=1
        else:
            num_spec = 5 - np.sum((all_num_card[6:11]>0).astype(np.float64))
            normal_card = np.where(all_num_card[6:11]>0)[0]+6
            special_card = np.where(all_num_card[0:6]>0)[0]
            for n in normal_card:
                type_card = n
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[57])[0][0]] = 6
                discard_pile[int(type_card)]+=1
            if num_spec<5:
                for i in range(int(num_spec)):
                    np.random.shuffle(special_card)
                    type_card = special_card[0]
                    env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[57])[0][0]] = 6
                    special_card = special_card[1:]
                    discard_pile[int(type_card)]+=1

    return env,discard_pile

@njit
def idPlayerCanUseNope(env,nope_id):
    """return the id of the player that have the nope card, else -1"""
    main_id = env[57]
    nope_turn = nopeTurn(main_id)
    idx_old = -1
    for i in range(4):
        if nope_turn[i] == nope_id:
            idx_old = i
            break
    else:
        idx_old = -1
    if idx_old+1==4:
        return main_id
    else:
        for i in range(idx_old+1,4):
            idx = nope_turn[i]
            if np.where(env[0:5]==idx)[0].shape[0]>=1 and env[62:67][int(idx)] == 1:
                return idx
        return main_id



@njit
def stepEnv(env,draw_pile,discard_pile,action):
    phase = env[67]
    main_id = env[57]
    nope_id = env[73]
    nope_count = env[56]
    last_action = env[72]
    if phase==0: #Phase 0: Main Turn
        if action==6: #draw card
            env,draw_pile,discard_pile = drawCard(env,draw_pile,discard_pile)
        else:
            env[72] = action
            if env[72]<=5:
                discardCardNormalAction(env,env[72],discard_pile)
            elif env[72]>=7:
                discardCardSpecialAction(env,env[72],discard_pile)

            env[73] = idPlayerCanUseNope(env,main_id)
            if env[73]==main_id:
                env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[72])
                #print(f'Action {env[72]} has been executed!')
            else:
                env[67] = 1 #change to Nope phase
    elif phase==1:#Phase 1: Nope phase

        if action==0 and env[73]!=main_id: #other player use Nope
            # print(f'Player {env[73]} use Nope!')
            env[56]+=1 # increase Nope Count
            env[0:5][np.where(env[0:5]==env[73])[0][0]] = 6
            discard_pile[0]+=1
            env[73] = idPlayerCanUseNope(env,env[73])
            if env[73]==main_id:
                if not checkIfNope(env): #if not been Nope
                    env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[72])
      
        elif action==0 and env[73]==main_id:
                env[56]+=1 # increase Nope Count
                env[0:5][np.where(env[0:5]==env[57])[0][0]] = 6
                if not checkIfNope(env): #if not been Nope
                    env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[72])
                    # print(f'Action {env[72]} has been executed!')
        else:
            if env[73]==main_id:
                if not checkIfNope(env): #if not been Nope
                    env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[72])
                    # print(f'Action {env[72]} has been executed!')
                else: # if Nope
                    if action==0:
                        # print('Main player use Yup!')
                        env[56] = 0 #reset to original
                        env[0:5][np.where(env[0:5]==main_id)[0][0]] = 6
                        discard_pile[0]+=1
                        env[73] = idPlayerCanUseNope(env,env[73])
                        if env[73]==main_id:
                            env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[72])
                            # print(f'Action {env[72]} has been executed!')
                    else:
                        # print(f'Action {env[72]} has been Nope!')
                        env[72] = -1# action has been Nope
                        env[67] = 0 # back to phase 0
                        env[73] = idPlayerCanUseNope(env,main_id)
            # env[73] = idPlayerCanUseNope(env,env[73])   
            else:
                env[73] = idPlayerCanUseNope(env,env[73])
                if env[73]==main_id:
                    if not checkIfNope(env): #if not been Nope
                        env,draw_pile,discard_pile = executeMainAction(env,draw_pile,discard_pile,env[72])
                
                        # print(f'Action {env[72]} has been executed!')
            
    elif phase==2:# phase 2: choose player to steal card. Only main_id can enter this phase
        if action==6:
            env,draw_pile,discard_pile = drawCard(env,draw_pile,discard_pile)
        else:
            env[74] = env[58:62][int(action-11)]
            last_action = env[72]
            #print(f'Player {env[57]} choose player {env[74]} to steal!')
            if last_action==7:
                card_on_player_chosen = np.where(env[0:56]==env[74])[0]
                card = np.random.choice(card_on_player_chosen)
                env[0:56][card] = env[57]
                #used card go to Discard Pile
                env[67] = 0
            else:
                env[67] = 3

    elif phase==3: #phase 3: choose card to give/take. Only main_id can enter this phase
        last_action = env[72]
        if last_action==3:
            type_card = action - 15
            all_card_to_take = np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[74])[0]
            env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][int(all_card_to_take[0])] = env[57]


        elif last_action==8:
            #take card
            type_card = action - 27
            all_card_to_take = np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==env[74])[0]
            if all_card_to_take.shape[0]>0:
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][int(all_card_to_take[0])] = env[57]
            #used card go to Discard Pile
        elif last_action==9:
            type_card = action - 39
            if np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==6)[0].shape[0]>0:
                env[getCardRange(type_card)[0]:getCardRange(type_card)[1]][np.where(env[getCardRange(type_card)[0]:getCardRange(type_card)[1]]==6)[0][0]] = env[57]
        last_action = -1
        env[67] = 0
    return env,draw_pile,discard_pile
@njit
def getAgentSize():
    return 5
@njit
def checkEnded(env):
    if np.sum(env[62:67])==1:
        return np.where(env[62:67]==1)[0][0]
    else:
        return -1
@njit
def getReward(state):
    if np.sum(state[82:86])==0:
        return 1
    elif state[86] == 0:
        return 0
    else:
        return -1

def visualCard(card):
    arr = []
    lst = ['Nope','Attack','Skip','Favor','Shuffle','See the future','TCT','RRC','BC','HPC','CTM','Defuse','Exploding Kitten']
    for i in card:
        arr.append(lst[int(getCardType(i))])
    return arr

def run(listAgent,perData):
    env,draw_pile,discard_pile = initEnv()
    for _ in range(getAgentSize()):
        dataOnePlayer = List()
        dataOnePlayer.append(np.array([[0.]]))
    winner = -1
    turn = 0
    while True:
        turn +=1
        phase = env[67]
        main_id = env[57]
        nope_id = env[73]
        last_action = env[72]
        if phase==0:
            pIdx = int(main_id)
        elif phase==1:
            pIdx = int(nope_id)
        elif phase==2:
            pIdx = int(main_id)
        elif phase==3:
            if last_action==3:
                pIdx = int(env[74])
            else:
                pIdx = int(main_id)

        action, perData = listAgent[pIdx](getAgentState(env,draw_pile,discard_pile), perData)
        env,draw_pile,discard_pile = stepEnv(env,draw_pile,discard_pile,action)
        winner = checkEnded(env)
        
        if winner != -1:
            break
    return winner, perData

def main(listAgent, times, perData):
    numWin = np.full(6, 0)
    pIdOrder = np.arange(5)
    for _ in range(times):
        np.random.shuffle(pIdOrder)
        shuffledListAgent = [listAgent[i] for i in pIdOrder]
        winner, perData = run(shuffledListAgent, perData)
        if winner == -1:
            numWin[-1] += 1
        else:
            numWin[pIdOrder[winner]] += 1
            #print(winner)
    return numWin, perData

@njit
def numbaRun(p0,p1,p2,p3,p4,perData,pIdOrder):
    env,draw_pile,discard_pile = initEnv()
    for _ in range(getAgentSize()):
        dataOnePlayer = List()
        dataOnePlayer.append(np.array([[0.]]))
    winner = -1
    turn = 0
    while True:
        turn +=1
        phase = env[67]
        main_id = env[57]
        nope_id = env[73]
        last_action = env[72]
        if phase==0:
            pIdx = int(main_id)
        elif phase==1:
            pIdx = int(nope_id)
        elif phase==2:
            pIdx = int(main_id)
        elif phase==3:
            if last_action==3:
                pIdx = int(env[74])
            else:
                pIdx = int(main_id)
        try:
            if pIdOrder[pIdx] == 0:
                action, perData = p0(getAgentState(env,draw_pile,discard_pile), perData)
            elif pIdOrder[pIdx] == 1:
                action, perData = p1(getAgentState(env,draw_pile,discard_pile), perData)
            elif pIdOrder[pIdx] == 2:
                action, perData = p2(getAgentState(env,draw_pile,discard_pile), perData)
            elif pIdOrder[pIdx] == 3:
                action, perData = p3(getAgentState(env,draw_pile,discard_pile), perData)
            elif pIdOrder[pIdx] == 4:
                action, perData = p4(getAgentState(env,draw_pile,discard_pile), perData)
            env,draw_pile,discard_pile = stepEnv(env,draw_pile,discard_pile,action)
        except:
            try:
                print(list(env),list(draw_pile),list(discard_pile),action)
            except:
                print(list(env),list(draw_pile),list(discard_pile))
            break
        
        winner = checkEnded(env)
        if winner != -1 or turn>150:
            break
    return winner, perData




@njit
def numbaMain(p0, p1, p2, p3,p4, times, perData):
    numWin = np.full(6, 0)
    pIdOrder = np.arange(5)
    for _ in range(times):
        np.random.shuffle(pIdOrder)
        winner, perData = numbaRun(p0, p1, p2, p3, p4, perData, pIdOrder)
        if winner == -1:
            numWin[5] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    return numWin, perData
@njit
def random_player(state,per):
    list_action  = np.where(getValidActions(state)==1)[0]
    action = np.random.choice(list_action)
    #print(state[82:86],state[87:91])
    return action,per
