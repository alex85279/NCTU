#include<pthread.h>
#include<cstdlib>
#include<stdio.h>
#include<iostream>
#include<sys/types.h>
#include<unistd.h>
#include<time.h>
#include<algorithm>
#include<mutex>
using namespace std;

bool m1_avail;
int m_played_round;
int m1_fin_time;
int m1_user_id;
int* line;
int* line_play_round;
int* fin_line1; 
int* line_result1;
int person_in_fin1;
int person_inline;
int now_time = 0;
int ness;
int person_num;
int ness_id;
int ness_m;
bool m_plus = false;

mutex gM1,gM2,gM3;

void* player(void *param);
struct player_info{
	int id;
	int arri_t;
	int play_round;
	int rest_time;
	int total_play;	
	bool finish;
};
struct player_info* p_inf;
int main(){
	cin>>ness;
	cin>>person_num;
	
	pthread_t tid[person_num];
	pthread_attr_t attr[person_num];

	line = new int[person_num];
	line_play_round = new int[person_num];
	fin_line1 = new int[person_num];
	line_result1 = new int[person_num];	
	p_inf = new struct player_info[person_num];
	person_inline = 0;
	person_in_fin1 = 0;
	m1_avail = true;
	m1_user_id = -1;
	ness_id = -1;
	ness_m = -1;
	m_played_round = -1;

	
	for(int i = 0; i<person_num;i++){
		
		cin>>p_inf[i].arri_t>>p_inf[i].play_round>>p_inf[i].rest_time>>p_inf[i].total_play;

	}
	for(int i = 0; i<person_num;i++){
		pthread_attr_init(&attr[i]);
		p_inf[i].id = i;
		p_inf[i].finish = false;
	}
	while(1){
		
		for(int i = 0; i<person_num;i++){
			if(p_inf[i].finish == false){
				pthread_create(&tid[i],&attr[i], player, (void*)&p_inf[i]);
			}
		}
		
		
		for(int i = 0; i<person_num;i++){
			pthread_join(tid[i],NULL);
		}
		
		if(m1_avail){
			m_played_round = -1;
		}
		
		//cout<<"player: "<<m1_user_id+1<<" mach played: "<<m_played_round<<endl;
		
		if(m1_user_id !=-1) p_inf[m1_user_id].total_play--;
		now_time++;
		int s_time = 0;
		for(int i = 0; i<person_num ;i++){
			if(p_inf[i].finish == true){
				s_time++;
			}
		}
		if(s_time == person_num ){
			break;
		}
		
	}
	
	
	cout<<endl<<"finish";
	
}

void *player(void *param){
	struct player_info *p_inf;
	p_inf = (struct player_info*) param;
	
	
	gM1.lock();
	if(p_inf->arri_t == now_time) {   // line up
		if(m1_avail == true && person_inline == 0){
			cout<<now_time<<" "<<p_inf->id+1<<" start playing"<<endl;
			m1_avail = false;
			m1_user_id = p_inf->id;
			m1_fin_time = now_time + p_inf->play_round;
			m_played_round++;
		}
		else{
			line[person_inline] =p_inf->id;
			line_play_round[person_inline] =p_inf->play_round;
			person_inline++;
			cout<<now_time<<" "<<p_inf->id+1<<" wait in line"<<endl;
		}	
	}	
	gM1.unlock();
	
	gM1.lock();
	if(p_inf->id == m1_user_id){
		if(p_inf->total_play== 0){
			cout<<now_time<<" "<<p_inf->id+1<<" finish playing YES"<<endl; 
			p_inf->finish = true;
			m1_avail = true;
			m1_user_id = -1;
			m_played_round = -1;
			//debug log
			//cout<<fin_line1[person_in_fin1]+1<<" "<<person_in_fin1<<endl;
		}
		else if(m_played_round == ness){
			cout<<now_time<<" "<<p_inf->id+1<<" finish playing YES"<<endl; 
			p_inf->finish = true;
			m1_avail = true;
			m1_user_id = -1;
			m_played_round = -1;
			//debug log
			//cout<<fin_line1[person_in_fin1-1]<<" "<<person_in_fin1<<endl;
		}
		else if(p_inf->id == m1_user_id && now_time == m1_fin_time){
			cout<<now_time<<" "<<p_inf->id+1<<" finish playing NO"<<endl;
			p_inf->arri_t = now_time + p_inf->rest_time;
			m1_avail = true;
			m1_user_id = -1;
			//debug log
			//cout<<fin_line1[person_in_fin1-1]<<" "<<person_in_fin1<<endl;
		}
		
		else{
			m_played_round++;
			//cout<<now_time<<" "<<m_played_round<<endl;
		}
	}

	while(m1_avail && person_inline>0){
		cout<<now_time<<" "<<line[0]+1<<" start playing"<<endl; 
		m1_user_id = line[0];
		m1_avail = false;
		m1_fin_time = now_time + line_play_round[0];
		for(int i = 0; i<person_inline-1;i++){
			line[i] = line[i+1];
			line_play_round[i] = line_play_round[i+1];
		}
		person_inline--;
		m_played_round++;
	//cout<<now_time<<" "<<m_played_round<<endl;
	}
	
	gM1.unlock();
	
	pthread_exit(0);
	
}
