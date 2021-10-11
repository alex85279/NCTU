#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<strings.h>
#include<string.h>
#include<json/json.h>
#include<stdio.h>
struct member{
	char id[100];
	char token[100];
	char passwd[100];
	int login;
	int post_num;
	int friend_num;
	int invite_num;
	char friend_list[100][100];
	char invite_list[100][100];
	char post[100][100];

}mem[100];
void delete_friend(int mem_idx,int friend_idx){
	for(int i = friend_idx;i<mem[mem_idx].friend_num;i++){
		strcpy(mem[mem_idx].friend_list[i],mem[mem_idx].friend_list[i+1]);
	}
	mem[mem_idx].friend_num--;
	
}
void delete_invite(int mem_idx,int friend_idx){
	for(int i = friend_idx;i<mem[mem_idx].invite_num;i++){
		strcpy(mem[mem_idx].invite_list[i],mem[mem_idx].invite_list[i+1]);
	}
	mem[mem_idx].invite_num--;
}
void arrange_mem(int idx,int mem_c){
	for(int i = 0; i<mem_c;i++){
		for(int j = 0; j<mem[i].friend_num;j++){
			if(strcmp(mem[i].friend_list[j],mem[idx].id)==0){
				delete_friend(i,j);
			}
		}
		for(int j = 0; j<mem[i].invite_num;j++){
			if(strcmp(mem[i].invite_list[j],mem[idx].id)==0){
				delete_invite(i,j);
			}
		}
	}
	for(int i = idx;i<mem_c;i++){
		memcpy(&mem[i],&mem[i+1],sizeof(mem[i]));
	
	}
	
	//memcpy(&mem[mem_c-1],0,sizeof(mem[mem_c-1]));
}
int main(){
	char ip[100];
	char port[100];
	char in;
	int tmp = 0;
	while((in = getchar())!='\n'){
		ip[tmp++] = in;
	}
	ip[tmp] = '\0';
	tmp = 0;
	while((in = getchar())!='\n'){
		port[tmp++] = in;
	}
	port[tmp] = '\0';
	int sockfd;
	int num;
	int mem_c;
	FILE *fp;
	fp = fopen("save.txt","r");
	char buff[255];
	fgets(buff,255,(FILE*)fp);
	strtok(buff,"\n");
	mem_c = atoi(buff);
	for(int i = 0;i<mem_c;i++){
		fgets(mem[i].id,100,(FILE*)fp); //id
		strtok(mem[i].id,"\n");
		

		fgets(mem[i].passwd,100,(FILE*)fp); //passwd
		strtok(mem[i].passwd,"\n");


		fgets(buff,100,(FILE*)fp); //login
		strtok(buff,"\n");
		mem[i].login = atoi(buff);

		fgets(buff,100,(FILE*)fp); //post_num
		strtok(buff,"\n");
		mem[i].post_num = atoi(buff);

		fgets(buff,100,(FILE*)fp); //friend_num
		strtok(buff,"\n");
		mem[i].friend_num = atoi(buff);

		fgets(buff,100,(FILE*)fp); //invite_num
		strtok(buff,"\n");
		mem[i].invite_num = atoi(buff);

		for(int j = 0; j<mem[i].friend_num;j++){   //friend_list
			fgets(mem[i].friend_list[j],100,(FILE*)fp);
			strtok(mem[i].friend_list[j],"\n");

		}
		for(int j = 0; j<mem[i].invite_num;j++){   //invite list
			fgets(mem[i].invite_list[j],100,(FILE*)fp);
			strtok(mem[i].invite_list[j],"\n");

		}

		for(int j = 0; j<mem[i].post_num;j++){
			fgets(mem[i].post[j],100,(FILE*)fp);
			strtok(mem[i].post[j],"\n");

		}
	}

	fclose(fp);
	printf("%s %s\n",ip,port);
	struct sockaddr_in dest;
	char buffer1[50];
	sockfd = socket(AF_INET,SOCK_STREAM,0);

	bzero(&dest,sizeof(dest));
	dest.sin_family = AF_INET;
	dest.sin_port = htons(atoi(port));
	dest.sin_addr.s_addr = inet_addr(ip);
	bind(sockfd,(struct sockaddr*)&dest, sizeof(dest));
	listen(sockfd,20);
	while(1){
		//param
		char* arg[20];
		char buffer2[100];
		num = 0;
		json_object *jobj = json_object_new_object();
		json_object *status;
		json_object *message;
		json_object *token;
		json_object *invite;
		json_object *friend;
		json_object *post;
		//connect and receive
		int clientfd;
		struct sockaddr_in client_addr;
		int addrlen = sizeof(client_addr);
		clientfd = accept(sockfd,(struct sockaddr*)&client_addr,&addrlen);
		
		int res = read(clientfd,buffer1,sizeof(buffer1));
		buffer1[res] = 0;
		strcpy(buffer2,buffer1);
		printf("receive from client: %s, %d bytes\n", buffer1,res);
		// parse the string
		for(int i=0;i<20;i++){
			arg[i] = NULL;
		}	
		num = 0;			
		char* pch;
		char* delim = " \n";
		pch = strtok(buffer1,delim);
		while(pch!=NULL){
			arg[num] = pch;
			pch = strtok(NULL,delim);
			num++;		
		} 
		//wrong command
		if(strcmp(arg[0],"register")!=0 && strcmp(arg[0],"login")!=0 && strcmp(arg[0],"logout")!=0 && strcmp(arg[0],"delete")!=0 &&  strcmp(arg[0],"invite")!=0 &&  strcmp(arg[0],"accept-invite")!=0 &&  strcmp(arg[0],"list-invite")!=0 &&  strcmp(arg[0],"list-friend")!=0 &&  strcmp(arg[0],"post")!=0 &&  strcmp(arg[0],"receive-post")!=0 ){
			message = json_object_new_string("unknown command");
			json_object_object_add(jobj,"message",message);
		}
		//command register
		if(strcmp(arg[0],"register")==0){
			int flg = 0;
			if(num>1){
				for(int i=0;i<mem_c;i++){
					if(strcmp(arg[1],mem[i].id)==0){
						flg = 1;
						break;
					}			
				}
			}
			if(num != 3){
				status = json_object_new_int(1);
				message = json_object_new_string("Usage: register <id> <password>");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else if(flg == 1){
				status = json_object_new_int(1);
				strcat(arg[1]," is already used");
				message = json_object_new_string(arg[1]);
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else{
				status = json_object_new_int(0);
				message = json_object_new_string("Success");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);


				strcpy(mem[mem_c].id,arg[1]);
				strcpy(mem[mem_c].passwd,arg[2]);
				mem[mem_c].login = 0;
				mem[mem_c].friend_num = 0;
				mem[mem_c].invite_num = 0;
				mem[mem_c].post_num = 0;
				mem_c++;
			}
			
		}
		
		//command login
		if(strcmp(arg[0],"login")==0){
			if(num != 3){
				status = json_object_new_int(1);
				message = json_object_new_string("Usage: login <id> <password>");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			int flg = 0;
			int idx;
			int pd_crt = 0;
			for(int i = 0; i<mem_c;i++){
				if(strcmp(arg[1],mem[i].id)==0){
					flg = 1;
					idx = i;
					if(strcmp(arg[2],mem[i].passwd)==0) pd_crt = 1;
					break;				
				}

			}
			
			if(flg == 0 || pd_crt == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("No such user or password error");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else{
				status = json_object_new_int(0);
				message = json_object_new_string("Success!");
				strcat(arg[1],"0000");
				token = json_object_new_string(arg[1]);
				printf("%s\n",json_object_to_json_string(token));
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				json_object_object_add(jobj,"token",token);
				
				mem[idx].login = 1;
				strcpy(mem[idx].token,arg[1]);
			}

		}

		//command delete
		if(strcmp(arg[0],"delete")==0){
			if(num == 1){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			int flg = 0;
			int idx;
			int log = 0;
			for(int i = 0; i<mem_c;i++){
				if(strcmp(arg[1],mem[i].token)==0){
					flg = 1;
					idx = i;
					log = mem[i].login;
					break;				
				}

			}
			if(num > 2){
				status = json_object_new_int(1);
				message = json_object_new_string("Usage: delete <user>");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else{
				status = json_object_new_int(0);
				message = json_object_new_string("Success!");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				arrange_mem(idx,mem_c);
				mem_c--;
				
			}

		}

		//command logout
		if(strcmp(arg[0],"logout")==0){
			
			int flg = 0;
			int idx;
			int log = 0;
			if(num > 1){
				for(int i = 0; i<mem_c;i++){
					if(strcmp(arg[1],mem[i].token)==0){
						flg = 1;
						idx = i;
						log = mem[i].login;
						printf("logout %s, id = %d\n",mem[i].id,idx);
						break;				
					}

				}
			}
			if(flg == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(log == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(num != 2 && flg == 1){
				status = json_object_new_int(1);
				message = json_object_new_string("Usage: logout <user>");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else{
				status = json_object_new_int(0);
				message = json_object_new_string("Bye!");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				mem[idx].login = 0;
				
			}
		
		}
		//command invite
		if(strcmp(arg[0],"invite")==0){
			
			int flg = 0;
			int idx;
			int log = 0;
			int exist = 0;
			int friend_idx;
			int already_friend = 0;
			int already_invite = 0;
			int is_invited = 0;
			int is_friend = 0;
			if(num>1){
				for(int i = 0; i<mem_c;i++){
					if(strcmp(arg[1],mem[i].token)==0){
						flg = 1;
						idx = i;
						log = mem[i].login;			
					}
					if(num>2){
						if(strcmp(arg[2],mem[i].id)==0){
							exist = 1;
							friend_idx = i;
						}
					}

				}
			}
			if(flg == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(num > 3 || num == 1 || (num == 2 && flg == 1)){
				status = json_object_new_int(1);
				message = json_object_new_string("Usage: invite <user> <id>");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(exist == 1){
				for(int i = 0; i<mem[idx].friend_num;i++){
					if(strcmp(mem[idx].friend_list[i],arg[2])==0){
						already_friend = 1;
						break;
					}
				}
				for(int i = 0; i<mem[idx].invite_num;i++){
					if(strcmp(mem[idx].invite_list[i],arg[2])==0){
						printf("is_invited = 1\n");
						is_invited = 1;
						break;
					}
				}
				
				for(int i = 0; i<mem[friend_idx].invite_num;i++){
					if(strcmp(mem[friend_idx].invite_list[i],mem[idx].id)==0){
						printf("already invite = 1\n");
						already_invite = 1;
							break;
					}
				}
			}
			if(exist == 0){
				status = json_object_new_int(1);
				strcat(arg[2]," does not exist");
				message = json_object_new_string(arg[2]);
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else if(already_friend == 1){
				status = json_object_new_int(1);
				strcat(arg[2]," is already your friend");
				message = json_object_new_string(arg[2]);
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else if(already_invite == 1){
				status = json_object_new_int(1);
				message = json_object_new_string("Already invited");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else if(is_invited == 1){
				status = json_object_new_int(1);
				strcat(arg[2]," has invited you");
				message = json_object_new_string(arg[2]);
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else if(friend_idx == idx){
				status = json_object_new_int(1);
				message = json_object_new_string("You cannot invite yourself");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			
			else{
				status = json_object_new_int(0);
				message = json_object_new_string("Success!");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				strcpy(mem[friend_idx].invite_list[mem[friend_idx].invite_num],mem[idx].id);
				mem[friend_idx].invite_num++;
				
			}
		}
		//command list-invite
		if(strcmp(arg[0],"list-invite")==0){
			printf("enter list-invite, num = %d\n",num);
			if(num == 1){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			int flg = 0;
			int idx;
			int log = 0;
			if(num>1){
				for(int i = 0; i<mem_c;i++){
					if(strcmp(arg[1],mem[i].token)==0){
						flg = 1;
						idx = i;
						log = mem[i].login;
						printf("id: %s, login: %d, id_num: %d\n",mem[i].id,log,idx);
						break;				
					}

				}
			}
			if(log == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(num > 2){
				status = json_object_new_int(1);
				message = json_object_new_string("Usage: list-invite <user>");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else{
				status = json_object_new_int(0);
				invite = json_object_new_array();
				for(int i = 0;i<mem[idx].invite_num;i++){
					json_object *inv = json_object_new_string(mem[idx].invite_list[i]);
					json_object_array_add(invite,inv);
				}
				json_object_object_add(jobj,"invite",invite);
				json_object_object_add(jobj,"status",status);
				
			}

		}
		//command accept-invite
		if(strcmp(arg[0],"accept-invite")==0){
			
			
			int flg = 0;
			int idx;
			int log = 0;
			int exist = 0;
			int inv_idx;
			int friend_idx;
			if(num > 1){
				for(int i = 0; i<mem_c;i++){
					if(strcmp(arg[1],mem[i].token)==0){
						flg = 1;
						idx = i;
						log = mem[i].login;			
					}
					if(num>2){
						if(strcmp(arg[2],mem[i].id)==0){
							friend_idx = i;
						}
					}

				}
			}
			if(num>2){
				for(int i = 0; i<mem[idx].invite_num;i++){
					if(strcmp(mem[idx].invite_list[i],arg[2])==0){
						exist = 1;
						inv_idx = i;
						break;
					}
				}
			}
			if(num !=3 && flg == 1){
				status = json_object_new_int(1);
				message = json_object_new_string("Usage: accept-invite <user> <id>");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(flg == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(log == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			
			else if(exist == 0){
				status = json_object_new_int(1);
				strcat(arg[2]," did not invite you");
				message = json_object_new_string(arg[2]);
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else{
				printf("idx = %d, inv-idx = %d",idx,inv_idx);
				status = json_object_new_int(0);
				message = json_object_new_string("Success!");
				json_object_object_add(jobj,"message",message);
				json_object_object_add(jobj,"status",status);
				delete_invite(idx,inv_idx);
				strcpy(mem[idx].friend_list[mem[idx].friend_num],arg[2]);
				mem[idx].friend_num++;
				strcpy(mem[friend_idx].friend_list[mem[friend_idx].friend_num],mem[idx].id);
				mem[friend_idx].friend_num++;
				
			}

		}
		//command list-friend
		if(strcmp(arg[0],"list-friend")==0){
			
			if(num == 1){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			int flg = 0;
			int idx;
			int log = 0;
			if(num>1){
				for(int i = 0; i<mem_c;i++){
					if(strcmp(arg[1],mem[i].token)==0){
						flg = 1;
						idx = i;
						log = mem[i].login;
						break;				
					}

				}
			}
			if(log == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(num > 2){
				status = json_object_new_int(1);
				message = json_object_new_string("Usage: list-friend <user>");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
			}
			else{
				status = json_object_new_int(0);
				friend = json_object_new_array();
				for(int i = 0;i<mem[idx].friend_num;i++){
					json_object *inv = json_object_new_string(mem[idx].friend_list[i]);
					json_object_array_add(friend,inv);
				}
				json_object_object_add(jobj,"friend",friend);
				json_object_object_add(jobj,"status",status);
				
			}

		}
		//command post
		if(strcmp(arg[0],"post")==0){
			
			
			int flg = 0;
			int idx;
			int log = 0;
			for(int i = 0; i<mem_c;i++){
				if(strcmp(arg[1],mem[i].token)==0){
					flg = 1;
					idx = i;
					log = mem[i].login;
					break;				
				}

			}
			if((num ==2 && flg == 1)||num == 1){
					
				status = json_object_new_int(1);
				message = json_object_new_string("Usage: post <user> <message>");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(log == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(num >= 2 && flg == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				
			}
			else{
				status = json_object_new_int(0);
				message = json_object_new_string("Success!");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				char* po;
				po = strtok(buffer2," ");
				po = strtok(NULL," ");
				po = strtok(NULL,"\0");
				printf("%s\n",po);
				/*for(int i = 2;i<num;i++){
					strcat(mem[idx].post[mem[idx].post_num],arg[i]);
					if(i!=num-1) strcat(mem[idx].post[mem[idx].post_num]," ");
				}*/
				strcpy(mem[idx].post[mem[idx].post_num],po);
				mem[idx].post_num++;
				
				
			}

		}
		printf("%d mem\n",mem_c);
		for(int i = 0; i<mem_c;i++){
			printf("%s %d\n",mem[i].id,i);
			printf("postnum = %d\n",mem[i].post_num);
			for(int j = 0; j<mem[i].post_num;j++){
				printf("%s\n",mem[i].post[j]);
			}
		}
		//command receive-post
		if(strcmp(arg[0],"receive-post")==0){
			
			if(num == 1){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			int flg = 0;
			int idx;
			int log = 0;
			for(int i = 0; i<mem_c;i++){
				if(strcmp(arg[1],mem[i].token)==0){
					flg = 1;
					idx = i;
					log = mem[i].login;
					break;				
				}

			}
			if(num>2 && flg == 1){
					
				status = json_object_new_int(1);
				message = json_object_new_string("Usage: receive-post <user>");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			if(flg == 0){
				status = json_object_new_int(1);
				message = json_object_new_string("Not login yet");
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"message",message);
				write(clientfd,json_object_to_json_string(jobj),200);
				continue;
			}
			else{	
				status = json_object_new_int(0);
				post = json_object_new_array();
				for(int i = 0;i<mem_c;i++){
					for(int j = 0; j<mem[idx].friend_num;j++){
						if(strcmp(mem[idx].friend_list[j],mem[i].id)==0){
							for(int k = 0 ;k<mem[i].post_num;k++){
								json_object *tmp_o = json_object_new_object();
								json_object *id = json_object_new_string(mem[i].id);
								json_object *mess = json_object_new_string(mem[i].post[k]);
								json_object_object_add(tmp_o,"id",id);
								json_object_object_add(tmp_o,"message",mess);
								json_object_array_add(post,tmp_o);
							}
						}					
					}
				}
				json_object_object_add(jobj,"status",status);
				json_object_object_add(jobj,"post",post);
				
			}

		}
		printf("%d mem\n",mem_c);
		for(int i = 0; i<mem_c;i++){
			printf("%s %d login: %d\n",mem[i].id,i,mem[i].login);
			printf("postnum = %d\n",mem[i].post_num);
			for(int j = 0; j<mem[i].post_num;j++){
				printf("%s\n",mem[i].post[j]);
			}
		}
		
		printf("%s\n",json_object_to_json_string(jobj));
		write(clientfd,json_object_to_json_string(jobj),200);


		char tmp_mem[10];
		char tmp_post_num[10];
		char tmp_friend_num[10];
		char tmp_invite_num[10];
		char tmp_login[10];
		fp = fopen("save.txt","w+");
		sprintf(tmp_mem,"%d",mem_c);
		fprintf(fp,"%s\n",tmp_mem);
		for(int i = 0; i<mem_c;i++){
			sprintf(tmp_post_num,"%d",mem[i].post_num);
			sprintf(tmp_friend_num,"%d",mem[i].friend_num);
			sprintf(tmp_invite_num,"%d",mem[i].invite_num);
			sprintf(tmp_login,"%d",mem[i].login);

			fprintf(fp,"%s\n",mem[i].id);
			fprintf(fp,"%s\n",mem[i].passwd);
			fprintf(fp,"0\n"); // login
			fprintf(fp,"%s\n",tmp_post_num);
			fprintf(fp,"%s\n",tmp_friend_num);
			fprintf(fp,"%s\n",tmp_invite_num);
			for(int j = 0; j<mem[i].friend_num;j++){
				fprintf(fp,"%s\n",mem[i].friend_list[j]);
			}
			for(int j = 0; j<mem[i].invite_num;j++){
				fprintf(fp,"%s\n",mem[i].invite_list[j]);
			}
			for(int j = 0; j<mem[i].post_num;j++){
				fprintf(fp,"%s\n",mem[i].post[j]);
			}
		


		}
		fclose(fp);


		close(clientfd);
		for(int i = 0; i<sizeof(buffer1);i++){
			buffer1[i] = 0;		
		}





	}	
	
	
	close(sockfd);
	return 0;
}
