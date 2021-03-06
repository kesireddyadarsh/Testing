//
//  main.cpp
//  HOF
//
//  Created by adarsh kesireddy on 6/19/17.
//  Copyright © 2017 AK. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <cassert>
#include <algorithm>
#include <stdio.h>
#include <fstream>
#include <string>
#include <sstream>
# include <math.h>
#include <list>




using namespace std;

bool run_simulation = true;
bool test_simulation = true;

#define PI 3.14159265


/*************************
 Neural Network
 ************************/

struct connect{
    double weight;
};

static double random_global(double a) { return a* (rand() / double(RAND_MAX)); }

// This is for each Neuron
class Neuron;
typedef vector<Neuron> Layer;

class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    vector<connect> z_outputWeights;
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    unsigned z_myIndex;
    double z_outputVal;
    void setOutputVal(double val) { z_outputVal = val; }
    double getOutputVal(void) const { return z_outputVal; }
    void feedForward(const Layer prevLayer);
    double transferFunction(double x);
    
};

//This creates connection with neurons.
Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c) {
        z_outputWeights.push_back(connect());
        z_outputWeights.back().weight = randomWeight() - 0.5;
    }
    z_myIndex = myIndex;
}

double Neuron::transferFunction(double x){
    
    int case_to_use = 1;
    switch (case_to_use) {
        case 1:
            return tanh(x);
            break;
        case 2:
            //Dont use this case
            return 1/(1+exp(x));
            break;
        case 3:
            return x/(1+abs(x));
            break;
            
        default:
            break;
    }
    
    return tanh(x);
}

void Neuron::feedForward(const Layer prevLayer){
    double sum = 0.0;
    bool debug_sum_flag = false;
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        if(debug_sum_flag == true){
            cout<<prevLayer[n].getOutputVal()<<endl;
            cout<<&prevLayer[n].z_outputWeights[z_myIndex];
            cout<<prevLayer[n].z_outputWeights[z_myIndex].weight;
        }
        sum += prevLayer[n].getOutputVal() * prevLayer[n].z_outputWeights[z_myIndex].weight;
        //cout<<"This is sum value"<<sum<<endl;
    }
    z_outputVal = Neuron::transferFunction(sum);
}

//This is single neural network
class Net{
public:
    Net(vector<unsigned> topology);
    void feedForward(vector<double> inputVals);
    vector<Layer> z_layer;
    vector<double> outputvaluesNN;
    double backProp();
    double z_error;
    double z_error_temp;
    vector<double> z_error_vector;
    void mutate();
    vector<double> temp_inputs;
    vector<double> temp_targets;
    
    //CCEA
    double fitness;
    vector<double> closest_dist_to_poi;
    vector<double> global_objective_values;
    vector<double> local_objective_values;
    vector<double> difference_objective_values;
    
    //For team
    
    int my_team_number;
    
    //For team
    double local_reward_wrt_team;
    double global_reward_wrt_team;
    double difference_reward_wrt_team;
    
    //Store x and y values
    vector<double> store_x_values;
    vector<double> store_y_values;
    
    //Testing CCEA
    bool for_testing = false;
    double best_value_so_far;
    
    //NSGA-II
    vector<int> dominating_over;
    double dominating_me;
    bool already_have_front;
    int front_number;
    vector<double> crowding_distance;
    double crowding_distance_rank;
    vector<vector<double>> front;
    vector<vector<double>> dominate;
    vector<double> num_donimated;
    double rank;
};

Net::Net(vector<unsigned> topology){
    
    for(int  numLayers = 0; numLayers<topology.size(); numLayers++){
        //unsigned numOutputs = numLayers == topology.size() - 1 ? 0 : topology[numLayers + 1];
        
        unsigned numOutputs;
        if (numLayers == topology.size()-1) {
            numOutputs=0;
        }else{
            numOutputs= topology[numLayers+1];
        }
        
        if(numOutputs>15){
            cout<<"Stop it number outputs coming out"<<numOutputs<<endl;
            exit(10);
        }
        
        z_layer.push_back(Layer());
        
        for(int numNeurons = 0; numNeurons <= topology[numLayers]; numNeurons++){
            //cout<<"This is neuron number:"<<numNeurons<<endl;
            z_layer.back().push_back(Neuron(numOutputs, numNeurons));
        }
    }
}

void Net::mutate(){
    /*
     //popVector[temp].z_layer[temp][temp].z_outputWeights[temp].weight
     */
    for (int l =0 ; l < z_layer.size(); l++) {
        for (int n =0 ; n< z_layer.at(l).size(); n++) {
            for (int z=0 ; z< z_layer.at(l).at(n).z_outputWeights.size(); z++) {
                z_layer.at(l).at(n).z_outputWeights.at(z).weight += random_global(.5)-random_global(.5);
            }
        }
    }
}

void Net::feedForward(vector<double> inputVals){
    
    assert(inputVals.size() == z_layer[0].size()-1);
    for (unsigned i=0; i<inputVals.size(); ++i) {
        z_layer[0][i].setOutputVal(inputVals[i]);
    }
    for (unsigned layerNum = 1; layerNum < z_layer.size(); ++layerNum) {
        Layer &prevLayer = z_layer[layerNum - 1];
        for (unsigned n = 0; n < z_layer[layerNum].size() - 1; ++n) {
            z_layer[layerNum][n].feedForward(prevLayer);
        }
    }
    temp_inputs.clear();
    
    
    Layer &outputLayer = z_layer.back();
    z_error_temp = 0.0;
    
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        //cout<<"This is value from outputlayer.getourputvalue:::::"<<outputLayer[n].getOutputVal()<<endl;
        //double delta = temp_targets[n] - outputLayer[n].getOutputVal();
        //cout<<"This is delta value::"<<delta;
        //z_error_temp += delta * delta;
        outputvaluesNN.push_back(outputLayer[n].getOutputVal());
    }
    
}

double Net::backProp(){
    z_error = 0.0;
    for (int temp = 0; temp< z_error_vector.size(); temp++) {
        //cout<<"This is z_error_vector"<<temp<<" value::"<< z_error_vector[temp]<<endl;
        z_error += z_error_vector[temp];
    }
    //    cout<<"This is z_error::"<<z_error<<endl;
    return z_error;
}

/***********************
 POI
 **********************/
class POI{
public:
    double x_position_poi,y_position_poi,value_poi;
    //Environment test;
    //vector<Rover> individualRover;
    vector<double> x_position_poi_vec;
    vector<double> y_position_poi_vec;
    vector<double> value_poi_vec;
};

/************************
 Environment
 ***********************/

class Environment{
public:
    vector<POI> individualPOI;
    vector<POI> group_1;
    vector<POI> group_2;
};

/************************
 Rover
 ***********************/

double resolve(double angle);


class Rover{
    //Environment environment_object;
public:
    double x_position,y_position;
    vector<double> x_position_vec,y_position_vec;
    vector<double> sensors;
    vector<Net> singleneuralNetwork;
    void sense_poi(double x, double y, double val);
    void sense_rover(double x, double y);
    double sense_poi_delta(double x_position_poi,double y_position_poi);
    double sense_rover_delta(double x_position_otherrover, double y_position_otherrover);
    vector<double> controls;
    double delta_x,delta_y;
    double theta;
    double phi;
    void reset_sensors();
    int find_quad(double x, double y);
    double find_phi(double x, double y);
    double find_theta(double x_sensed, double y_sensed);
    void move_rover(double dx, double dy);
    double reward =0.0;
    void sense_all_values(vector<double> x_position_poi_vec_rover,vector<double> y_position_poi_vec_rover,vector<double> value_poi_vec_rover);
    void sense_values(vector<double> x_position_poi_vec_rover,vector<double> y_position_poi_vec_rover,vector<double> value_poi_vec_rover,vector<int>* p_index_number,int current_number,vector<Rover>* teamRover);
    double sense_rover_new(double current_x, double current_y, double x_position_otherrover, double y_position_otherrover);
    double sense_poi_new(double xposition, double yposition,double x_position_poi,double y_position_poi );
    int find_quad_new(double x,double y, double the, double x_sense, double y_sense);
    //stored values
    vector<double> max_reward;
    vector<double> policy;
    //vector<double> best_closest_distance;
    
    //Neural network
    vector<Net> network_for_agent;
    void create_neural_network_population(int numNN,vector<unsigned> topology);
    
    //random numbers for neural networks
    vector<int> random_numbers;
    
};

// variables used: indiNet -- object to Net
void Rover::create_neural_network_population(int numNN,vector<unsigned> topology){
    
    for (int populationNum = 0 ; populationNum<numNN; populationNum++) {
        //cout<<"This is neural network:"<<populationNum<<endl;
        Net singleNetwork(topology);
        network_for_agent.push_back(singleNetwork);
    }
    
}

//Function returns: sum of values of POIs divided by their distance
double Rover::sense_poi_delta(double x_position_poi,double y_position_poi ){
    double delta_sense_poi=0;
    double distance = sqrt(pow(x_position-x_position_poi, 2)+pow(y_position-y_position_poi, 2));
    double minimum_observation_distance =0.0;
    delta_sense_poi=(distance>minimum_observation_distance)?distance:minimum_observation_distance ;
    return delta_sense_poi;
}

double Rover::sense_poi_new(double xposition, double yposition,double x_position_poi,double y_position_poi ){
    double delta_sense_poi=0;
    double distance = sqrt(pow(xposition-x_position_poi, 2)+pow(yposition-y_position_poi, 2));
    double minimum_observation_distance =0.0;
    delta_sense_poi=(distance>minimum_observation_distance)?distance:minimum_observation_distance ;
    return delta_sense_poi;
}

//Function returns: sum of sqaure distance from a rover to all the other rovers in the quadrant
double Rover::sense_rover_delta(double x_position_otherrover, double y_position_otherrover){
    double delta_sense_rover=0.0;
    if (x_position_otherrover == NULL || y_position_otherrover == NULL) {
        return delta_sense_rover;
    }
    double distance = sqrt(pow(x_position-x_position_otherrover, 2)+pow(y_position-y_position_otherrover, 2));
    delta_sense_rover=(1/distance);
    
    return delta_sense_rover;
}

double Rover::sense_rover_new(double current_x, double current_y, double x_position_otherrover, double y_position_otherrover){
    double delta_sense_rover=0.0;
    double distance = sqrt(pow(current_x-x_position_otherrover, 2)+pow(current_y-y_position_otherrover, 2));
    delta_sense_rover=(1/distance);
    
    return delta_sense_rover;
}

void Rover::sense_poi(double poix, double poiy, double val){
    double delta = sense_poi_delta(poix, poiy);
    int quad = find_quad(poix,poiy);
    sensors.at(quad) += val/delta;
}

void Rover::sense_rover(double otherx, double othery){
    double delta = sense_rover_delta(otherx,othery);
    int quad = find_quad(otherx,othery);
    sensors.at(quad+4) += 1/delta;
}

void Rover::reset_sensors(){
    sensors.clear();
    for(int i=0; i<8; i++){
        sensors.push_back(0.0);
    }
}

double Rover::find_phi(double x_sensed, double y_sensed){
    double distance_in_x_phi =  x_sensed - x_position;
    double distance_in_y_phi =  y_sensed - y_position;
    double deg2rad = 180/PI;
    double phi = (atan2(distance_in_x_phi,distance_in_y_phi) *(deg2rad));
    
    return phi;
}

double Rover::find_theta(double x_sensed, double y_sensed){
    double distance_in_x_theta =  x_sensed - x_position;
    double distance_in_y_theta =  y_sensed - y_position;
    theta += atan2(distance_in_x_theta,distance_in_y_theta) * (180 / PI);
    
    return phi;
}

int Rover::find_quad_new(double x,double y, double the, double x_sense, double y_sense){
    int quadrant;
    
    double distance_in_x_phi =  x_sense - x;
    double distance_in_y_phi =  y_sense - y;
    double deg2rad = 180/PI;
    double phi = (atan2(distance_in_x_phi,distance_in_y_phi) *(deg2rad));
    double quadrant_angle = phi - the;
    
    quadrant_angle = resolve(quadrant_angle);
    assert(quadrant_angle != NAN);
    //    cout << "IN QUAD: FIND PHI: " << phi << endl;
    
    phi = resolve(phi);
    
    //    cout << "IN QUAD: FIND PHI2: " << phi << endl;
    
    int case_number;
    if ((0 <= quadrant_angle && 45 >= quadrant_angle)||(315 < quadrant_angle && 360 >= quadrant_angle)) {
        //do something in Q1
        case_number = 0;
    }else if ((45 < quadrant_angle && 135 >= quadrant_angle)) {
        // do something in Q2
        case_number = 1;
    }else if((135 < quadrant_angle && 225 >= quadrant_angle)){
        //do something in Q3
        case_number = 2;
    }else if((225 < quadrant_angle && 315 >= quadrant_angle)){
        //do something in Q4
        case_number = 3;
    }
    quadrant = case_number;
    
    //    cout << "QUADANGLE =  " << quadrant_angle << endl;
    //    cout << "QUADRANT = " << quadrant << endl;
    
    return quadrant;
}

int Rover::find_quad(double x_sensed, double y_sensed){
    int quadrant;
    
    double phi = find_phi(x_sensed, y_sensed);
    double quadrant_angle = phi - theta;
    quadrant_angle = resolve(quadrant_angle);
    assert(quadrant_angle != NAN);
    //    cout << "IN QUAD: FIND PHI: " << phi << endl;
    
    phi = resolve(phi);
    
    //    cout << "IN QUAD: FIND PHI2: " << phi << endl;
    
    int case_number;
    if ((0 <= quadrant_angle && 45 >= quadrant_angle)||(315 < quadrant_angle && 360 >= quadrant_angle)) {
        //do something in Q1
        case_number = 0;
    }else if ((45 < quadrant_angle && 135 >= quadrant_angle)) {
        // do something in Q2
        case_number = 1;
    }else if((135 < quadrant_angle && 225 >= quadrant_angle)){
        //do something in Q3
        case_number = 2;
    }else if((225 < quadrant_angle && 315 >= quadrant_angle)){
        //do something in Q4
        case_number = 3;
    }
    quadrant = case_number;
    
    //    cout << "QUADANGLE =  " << quadrant_angle << endl;
    //    cout << "QUADRANT = " << quadrant << endl;
    
    return quadrant;
}

void Rover::move_rover(double dx, double dy){
    
    double aom = atan2(dy,dx)*180/PI; /// angle of movement
    double rad2deg = PI/180;
    x_position = x_position + sin(theta*rad2deg) * dy + cos(theta*rad2deg) * dx;
    y_position = y_position + sin(theta*rad2deg) * dx + cos(theta*rad2deg) * dy;
    theta = theta + aom;
    theta = resolve(theta);
    
    //x_position =(x_position)+  (dy* cos(theta*(PI/180)))-(dx *sin(theta*(PI/180)));
    //y_position =(y_position)+ (dy* sin(theta*(PI/180)))+(dx *cos(theta*(PI/180)));
    //theta = theta+ (atan2(dx,dy) * (180 / PI));
    //theta = resolve(theta);
}


//Takes all poi values and update sensor values
void Rover::sense_all_values(vector<double> x_position_poi_vec_rover,vector<double> y_position_poi_vec_rover,vector<double> value_poi_vec_rover){
    reset_sensors();
    
    double temp_delta_value = 0.0;
    vector<double> temp_delta_vec;
    int temp_quad_value =0;
    vector<double> temp_quad_vec;
    
    assert(x_position_poi_vec_rover.size() == y_position_poi_vec_rover.size());
    assert(value_poi_vec_rover.size() == y_position_poi_vec_rover.size());
    
    for (int value_calculating_delta = 0 ; value_calculating_delta < x_position_poi_vec_rover.size(); value_calculating_delta++) {
        temp_delta_value = sense_poi_delta(x_position_poi_vec_rover.at(value_calculating_delta), y_position_poi_vec_rover.at(value_calculating_delta));
        temp_delta_vec.push_back(temp_delta_value);
    }
    
    for (int value_calculating_quad = 0 ; value_calculating_quad < x_position_poi_vec_rover.size(); value_calculating_quad++) {
        temp_quad_value = find_quad(x_position_poi_vec_rover.at(value_calculating_quad), y_position_poi_vec_rover.at(value_calculating_quad));
        temp_quad_vec.push_back(temp_quad_value);
    }
    
    assert(temp_delta_vec.size()== temp_quad_vec.size());
    
    for (int update_sensor = 0 ; update_sensor<temp_quad_vec.size(); update_sensor++) {
        sensors.at(temp_quad_vec.at(update_sensor)) += value_poi_vec_rover.at(update_sensor)/temp_delta_vec.at(update_sensor);
    }
    
}


void Rover::sense_values(vector<double> x_position_poi_vec_rover,vector<double> y_position_poi_vec_rover,vector<double> value_poi_vec_rover,vector<int>* p_index_number,int current_number,vector<Rover>* teamRover){
    
    //First sense POIs
    double temp_delta_value = 0.0;
    vector<double> temp_delta_vec;
    int temp_quad_value =0;
    vector<double> temp_quad_vec;
    
    assert(x_position_poi_vec_rover.size() == y_position_poi_vec_rover.size());
    assert(value_poi_vec_rover.size() == y_position_poi_vec_rover.size());
    
    for (int value_calculating_delta = 0 ; value_calculating_delta < x_position_poi_vec_rover.size(); value_calculating_delta++) {
        temp_delta_value = sense_rover_new(teamRover->at(current_number).x_position, teamRover->at(current_number).y_position,x_position_poi_vec_rover.at(value_calculating_delta), y_position_poi_vec_rover.at(value_calculating_delta));
        temp_delta_vec.push_back(temp_delta_value);
    }
    
    for (int value_calculating_quad = 0 ; value_calculating_quad < x_position_poi_vec_rover.size(); value_calculating_quad++) {
        temp_quad_value = find_quad_new(teamRover->at(current_number).x_position, teamRover->at(current_number).y_position, teamRover->at(current_number).theta, x_position_poi_vec_rover.at(value_calculating_quad), y_position_poi_vec_rover.at(value_calculating_quad));
        temp_quad_vec.push_back(temp_quad_value);
    }
    
    assert(temp_delta_vec.size()== temp_quad_vec.size());
    
    //Now sense rovers
    double new_temp_delta_value = 0.0;
    vector<double> new_temp_delta_vec;
    int new_temp_quad_value =0;
    vector<double> new_temp_quad_vec;
    
    for (int other_rover = 0; other_rover < p_index_number->size(); other_rover++) {
        if (current_number != other_rover) {
            //x and y coordinates
            new_temp_delta_value = sense_rover_new(teamRover->at(current_number).x_position, teamRover->at(current_number).y_position, teamRover->at(other_rover).x_position, teamRover->at(other_rover).y_position);
            new_temp_delta_vec.push_back(new_temp_delta_value);
            new_temp_quad_value = find_quad_new(teamRover->at(current_number).x_position, teamRover->at(current_number).y_position, teamRover->at(current_number).theta,teamRover->at(other_rover).x_position, teamRover->at(other_rover).y_position);
            new_temp_quad_vec.push_back(new_temp_quad_value);
        }
    }
    
    assert(new_temp_delta_vec.size()== new_temp_quad_vec.size());
    
    for (int update_sensor = 0 ; update_sensor<temp_quad_vec.size(); update_sensor++) {
        sensors.at(temp_quad_vec.at(update_sensor)) += value_poi_vec_rover.at(update_sensor)/temp_delta_vec.at(update_sensor);
    }
    
    for (int update_sensor = 0 ; update_sensor<new_temp_quad_vec.size(); update_sensor++) {
        sensors.at(new_temp_quad_vec.at(update_sensor)) += new_temp_delta_vec.at(update_sensor);
    }
}


/*************************
 Population
 ************************/
//This is for population of neural network
class Population{
public:
    void create_Population(int numNN,vector<unsigned> topology);
    vector<Net> popVector;
    void runNetwork(vector<double> inputVal,int number_neural);
    void sortError();
    void mutation(int numNN);
    void newerrorvector();
    void findindex();
    int returnIndex(int numNN);
    void repop(int numNN);
    
};

// variables used: indiNet -- object to Net
void Population::create_Population(int numNN,vector<unsigned> topology){
    
    for (int populationNum = 0 ; populationNum<numNN; populationNum++) {
        //cout<<"This is neural network:"<<populationNum<<endl;
        Net singleNetwork(topology);
        popVector.push_back(singleNetwork);
    }
    
}

//Return index of higher
int Population::returnIndex(int numNN){
    int temp = numNN;
    int number_1 = (rand() % temp);
    int number_2 = (rand() % temp);
    while (number_1 == number_2) {
        number_2 = (rand() % temp);
    }
    
    if (popVector[number_1].z_error<popVector[number_2].z_error) {
        return number_2;
    }else if (popVector[number_1].z_error>popVector[number_2].z_error){
        return number_1;
    }else{
        return NULL;
    }
}

void Population::repop(int numNN){
    for (int temp =0 ; temp<numNN/2; temp++) {
        int R = rand()% popVector.size();
        popVector.push_back(popVector.at(R));
        popVector.back().mutate();
    }
}

void Population::runNetwork(vector<double> inputVals,int num_neural){
    popVector.at(num_neural).feedForward(inputVals);
    popVector.at(num_neural).backProp();
}

/**************************
 Simulation Functions
 **************************/
// Will resolve angle between 0 to 360
double resolve(double angle){
    while(angle >= 360){
        angle -=360;
    }
    while(angle < 0){
        angle += 360;
    }
    while (angle == 360) {
        angle = 0;
    }
    return angle;
}


double find_scaling_number(vector<Rover>* teamRover, POI* individualPOI){
    double number =0.0;
    double temp_number =0.0;
    vector < vector <double> > group_sensors;
//    vector<double> xposition;
//    vector<double> yposition;
//    Rover R_obj; //Rover object
//    POI P_obj; // POI object
//    
//    P_obj.x_position_poi=50.0;
//    P_obj.y_position_poi=100.0;
//    P_obj.x_position_poi=150.0;
//    P_obj.y_position_poi=50.0;
//    P_obj.value_poi = 50;
//    P_obj.value_poi =100;
//    
//    int temp_rand = rand()%100;
//    while (temp_rand == 0) {
//        temp_rand = rand()%100;
//    }

    
//    //for (int temp_rover = 0; temp_rover < 2; temp_rover++) {
//        for (int temp=0; temp<temp_rand; temp++) {
//            R_obj.x_position=rand()%100;
//            R_obj.y_position=rand()%100;
//            xposition.push_back(R_obj.x_position);
//            yposition.push_back(R_obj.y_position);
//            R_obj.reset_sensors();
//            R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
//            group_sensors.push_back(R_obj.sensors);
//        }
//    //}
    
    for (int rover_number =0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy_number = 0; policy_number< individualPOI->value_poi_vec.size(); policy_number++) {
            teamRover->at(rover_number).reset_sensors();
            teamRover->at(rover_number).sense_poi(individualPOI->x_position_poi_vec.at(policy_number), individualPOI->y_position_poi_vec.at(policy_number), individualPOI->value_poi_vec.at(policy_number));
            group_sensors.push_back(teamRover->at(rover_number).sensors);
        }
    }
    
    assert(!group_sensors.empty());
    
    for (int i=0; i<group_sensors.size(); i++) {
        temp_number=*max_element(group_sensors.at(i).begin(), group_sensors.at(i).end());
        if (temp_number>number) {
            number=temp_number;
        }
    }
    
//    R_obj.reset_sensors();
    
    assert(number != 0.0);
//    xposition.clear();
//    yposition.clear();
    return number;
}

/*
void remove_lower_fitness_network(Population * p_Pop,vector<Rover>* p_rover){
    
    bool VERBOSE = false;
    
    //evolution
    double temp_selection_number= p_Pop->popVector.size()/2; //select half the size
    for (int selectNN=0; selectNN<(temp_selection_number); ++selectNN) {
        double temp_random_1 = rand()%p_Pop->popVector.size();
        double temp_random_2 = rand()%p_Pop->popVector.size();
        while(temp_random_1==temp_random_2) {
            temp_random_2 = rand()%p_Pop->popVector.size();
        }
        double random_rover_number = rand()%p_rover->size();
        
        if (p_rover->at(random_rover_number).max_reward.at(temp_random_1)>p_rover->at(random_rover_number).max_reward.at(temp_random_2)) {
            //delete neural network temp_random_2
            p_Pop->popVector.erase(p_Pop->popVector.begin()+temp_random_2);
            p_rover->at(random_rover_number).max_reward.erase(p_rover->at(random_rover_number).max_reward.begin()+temp_random_2);
        }else{
            //delete neural network temp_random_1
            p_Pop->popVector.erase(p_Pop->popVector.begin()+temp_random_1);
            p_rover->at(random_rover_number).max_reward.erase(p_rover->at(random_rover_number).max_reward.begin()+temp_random_1);
        }
        
        
    }
    
    //clear maximum values
    for (int clear_max_vec =0 ; clear_max_vec<p_rover->size(); clear_max_vec++) {
        p_rover->at(clear_max_vec).max_reward.clear();
    }
    
    if(VERBOSE){
        cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$\n\n"<<endl;
        for (int temp_print =0 ; temp_print<p_rover->size(); temp_print++) {
            cout<<"This is size of max reward::"<<p_rover->at(temp_print).max_reward.size()<<endl;
        }
    }
}

void repopulate_neural_networks(int numNN,Population* p_Pop){
    vector<unsigned> a;
    for (int temp =0 ; temp<numNN/2; temp++) {
        int R = rand()% p_Pop->popVector.size();
        Net N(a);
        N=p_Pop->popVector.at(R);
        N.mutate();
        p_Pop->popVector.push_back(N);
    }
}

/*****************************************************************
 Test Rover in environment
 ***************************************************************/

// Tests Stationary POI and Stationary Rover in all directions
bool POI_sensor_test(){
    bool VERBOSE = false;
    
    bool passfail = false;
    
    bool pass1 = false;
    bool pass2 = false;
    bool pass3 = false;
    bool pass4 = false;
    
    POI P;
    Rover R;
    
    /// Stationary Rover
    R.x_position = 0;
    R.y_position = 0;
    R.theta = 0; /// north
    
    P.value_poi = 10;
    
    /// POI directly north, sensor 0 should read; no others.
    P.x_position_poi = 0.001;
    P.y_position_poi = 1;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) != 0 && R.sensors.at(1) == 0 && R.sensors.at(2) ==0 && R.sensors.at(3) == 0){
        pass1 = true;
    }
    
    assert(pass1 == true);
    
    if(VERBOSE){
        cout << "Direct north case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    
    /// POI directly south, sensor 2 should read; no others.
    P.x_position_poi = 0;
    P.y_position_poi = -1;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) == 0 && R.sensors.at(1) == 0 && R.sensors.at(2) !=0 && R.sensors.at(3) == 0){
        pass2 = true;
    }
    
    assert(pass2 == true);
    
    if(VERBOSE){
        cout << "Direct south case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    
    /// POI directly east, sensor 1 should read; no others.
    P.x_position_poi = 1;
    P.y_position_poi = 0;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) == 0 && R.sensors.at(1) != 0 && R.sensors.at(2) ==0 && R.sensors.at(3) == 0){
        pass3 = true;
    }
    
    assert(pass3 == true);
    
    if(VERBOSE){
        cout << "Direct east case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    
    
    /// POI directly west, sensor 3 should read; no others.
    P.x_position_poi = -1;
    P.y_position_poi = 0;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) == 0 && R.sensors.at(1) == 0 && R.sensors.at(2) ==0 && R.sensors.at(3) != 0){
        pass4 = true;
    }
    
    if(VERBOSE){
        cout << "Direct west case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    assert(pass4 == true);
    
    
    if(pass1 && pass2 && pass3 && pass4){
        passfail = true;
    }
    assert(passfail == true);
    return passfail;
}

//Test for stationary rovers test in all directions
bool rover_sensor_test(){
    bool passfail = false;
    
    bool pass5 = false;
    bool pass6 = false;
    bool pass7 = false;
    bool pass8 = false;
    
    Rover R1;
    Rover R2;
    R1.x_position = 0;
    R1.y_position = 0;
    R1.theta = 0; // north
    R2.theta = 0;
    
    // case 1, Rover 2 to the north
    R2.x_position = 0;
    R2.y_position = 1;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 4 should fire, none other.
    if(R1.sensors.at(4) != 0 && R1.sensors.at(5) == 0 && R1.sensors.at(6) == 0 && R1.sensors.at(7) == 0){
        pass5 = true;
    }
    assert(pass5 == true);
    
    // case 2, Rover 2 to the east
    R2.x_position = 1;
    R2.y_position = 0;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 5 should fire, none other.
    if(R1.sensors.at(4) == 0 && R1.sensors.at(5) != 0 && R1.sensors.at(6) == 0 && R1.sensors.at(7) == 0){
        pass6 = true;
    }
    assert(pass6 == true);
    
    // case 3, Rover 2 to the south
    R2.x_position = 0;
    R2.y_position = -1;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 6 should fire, none other.
    if(R1.sensors.at(4) == 0 && R1.sensors.at(5) == 0 && R1.sensors.at(6) != 0 && R1.sensors.at(7) == 0){
        pass7 = true;
    }
    assert(pass7 == true);
    
    // case 4, Rover 2 to the west
    R2.x_position = -1;
    R2.y_position = 0;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 7 should fire, none other.
    if(R1.sensors.at(4) == 0 && R1.sensors.at(5) == 0 && R1.sensors.at(6) == 0 && R1.sensors.at(7) != 0){
        pass8 = true;
    }
    assert(pass8 == true);
    
    if(pass5 && pass6 && pass7 && pass8){
        passfail = true;
    }
    assert(passfail == true);
    return passfail;
}

void custom_test(){
    Rover R;
    POI P;
    R.x_position = 0;
    R.y_position = 0;
    R.theta = 90;
    
    P.x_position_poi = 0.56;
    P.y_position_poi = -1.91;
    P.value_poi = 100;
    
    R.reset_sensors();
    R.sense_poi(P.x_position_poi,P.y_position_poi,P.value_poi);
    
    
}

//x and y position of poi
vector< vector <double> > poi_positions;
vector<double> poi_positions_loc;

void stationary_rover_test(double x_start,double y_start){//Pass x_position,y_position
    Rover R_obj; //Rover object
    POI P_obj;
    
    R_obj.reset_sensors();
    
    //x and y position of poi
    vector< vector <double> > poi_positions;
    vector<double> poi_positions_loc;
    
    R_obj.x_position =x_start;
    R_obj.y_position=y_start;
    R_obj.theta=0.0;
    int radius = 2;
    
    double angle=0;
    
    P_obj.value_poi=100;
    
    int quad_0=0,quad_1=0,quad_2=0,quad_3=0,quad_0_1=0;
    while (angle<360) {
        if ((0<=angle && 45>= angle)) {
            quad_0++;
        }else if ((45<angle && 135>= angle)) {
            // do something in Q2
            quad_1++;
        }else if((135<angle && 225>= angle)){
            //do something in Q3
            quad_2++;
        }else if((225<angle && 315>= angle)){
            //do something in Q4
            quad_3++;
        }else if ((315<angle && 360> angle)){
            quad_0_1++;
        }
        poi_positions_loc.push_back(R_obj.x_position+(radius*cos(angle * (PI /180))));
        poi_positions_loc.push_back(R_obj.y_position+(radius*sin(angle * (PI /180))));
        poi_positions.push_back(poi_positions_loc);
        poi_positions_loc.clear();
        angle+=7;
    }
    
    vector<bool> checkPass_quad_1,checkPass_quad_2,checkPass_quad_3,checkPass_quad_0;
    
    for (int i=0; i<poi_positions.size(); i++) {
        for (int j=0; j<poi_positions.at(i).size(); j++) {
            P_obj.x_position_poi = poi_positions.at(i).at(j);
            P_obj.y_position_poi = poi_positions.at(i).at(++j);
            R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
            if (R_obj.sensors.at(0) != 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
                checkPass_quad_0.push_back(true);
            }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) != 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0){
                checkPass_quad_1.push_back(true);
            }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) !=0 && R_obj.sensors.at(3) == 0){
                checkPass_quad_2.push_back(true);
            }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) != 0){
                checkPass_quad_3.push_back(true);
            }
            R_obj.reset_sensors();
        }
    }
    if (checkPass_quad_0.size() != (quad_0_1+quad_0)) {
        cout<<"Something wrong with quad_0"<<endl;;
    }else if (checkPass_quad_1.size() != (quad_1)){
        cout<<"Something wrong with quad_1"<<endl;
    }else if (checkPass_quad_2.size() != quad_2){
        cout<<"Something wrong with quad_2"<<endl;
    }else if (checkPass_quad_3.size() != quad_3){
        cout<<"Something wrong with quad_3"<<endl;
    }
}

void find_x_y_stationary_rover_test_1(double angle, double radius, double x_position, double y_position){
    poi_positions_loc.push_back(x_position+(radius*cos(angle * (PI /180))));
    poi_positions_loc.push_back(y_position+(radius*sin(angle * (PI /180))));
}

void stationary_rover_test_1(double x_start,double y_start){
    bool VERBOSE = false;
    Rover R_obj; //Rover object
    POI P_obj;
    
    R_obj.reset_sensors();
    
    R_obj.x_position =x_start;
    R_obj.y_position=y_start;
    R_obj.theta=0.0;
    int radius = 2;
    
    bool check_pass = false;
    
    double angle=0;
    
    P_obj.value_poi=100;
    
    while (angle<360) {
        find_x_y_stationary_rover_test_1(angle, radius, R_obj.x_position, R_obj.y_position);
        P_obj.x_position_poi = poi_positions_loc.at(0);
        P_obj.y_position_poi = poi_positions_loc.at(1);
        R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
        if (R_obj.sensors.at(0) != 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 0"<<endl;
            }
            check_pass = true;
        }else  if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) != 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 1"<<endl;
                
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) !=0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 2"<<endl;
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) != 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 3"<<endl;
            }
            check_pass = true;
        }else{
            cout<<"Issue at an angle ::"<<angle<<" with x_position and y_position"<<R_obj.x_position<<R_obj.y_position<<endl;
            exit(10);
        }
        assert(check_pass==true);
        poi_positions_loc.clear();
        R_obj.reset_sensors();
        angle+=7;
        check_pass=false;
    }
}

void stationary_poi_test(double x_start,double y_start){
    bool VERBOSE = false;
    Rover R_obj; //Rover object
    POI P_obj; // POI object
    vector<double> rover_position_loc;
    
    R_obj.reset_sensors();
    
    P_obj.x_position_poi=x_start;
    P_obj.y_position_poi=y_start;
    P_obj.value_poi=100;
    R_obj.theta=0.0;
    
    R_obj.x_position =0.0;
    R_obj.y_position =0.0;
    
    bool check_pass = false;
    
    for (int i=0; i<=R_obj.theta; ) {
        if (R_obj.theta > 360) {
            break;
        }
        R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
        if (VERBOSE) {
            cout<<endl;
            for (int j=0; j<R_obj.sensors.size(); j++) {
                cout<<R_obj.sensors.at(j)<<"\t";
            }
            cout<<endl;
        }
        if (R_obj.sensors.at(0) != 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 0"<<endl;
            }
            check_pass = true;
        }else  if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) != 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 1";
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) !=0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 2";
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) != 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 3";
            }
            check_pass = true;
        }else{
            cout<<"Issue at an angle ::"<<R_obj.theta<<" with x_position and y_position"<<P_obj.x_position_poi<<P_obj.y_position_poi<<endl;
            exit(10);
        }
        assert(check_pass==true);
        i+=7;
        R_obj.theta+=7;
        R_obj.reset_sensors();
    }
}

void two_rovers_test(double x_start, double y_start){
    bool VERBOSE = false;
    Rover R_obj; //Rover object
    POI P_obj; // POI object
    vector<double> rover_position_loc;
    
    R_obj.reset_sensors();
    
    double otherRover_x = x_start;
    double otherRover_y = y_start;
    P_obj.value_poi=100;
    R_obj.theta=0.0;
    
    R_obj.x_position =0.0;
    R_obj.y_position =0.0;
    
    bool check_pass = false;
    
    for (int i=0; i<=R_obj.theta; ) {
        if (R_obj.theta > 360) {
            break;
        }
        R_obj.sense_rover(otherRover_x, otherRover_y);
        if (VERBOSE) {
            cout<<endl;
            for (int j=0; j<R_obj.sensors.size(); j++) {
                cout<<R_obj.sensors.at(j)<<"\t";
            }
            cout<<endl;
        }
        if (R_obj.sensors.at(4) != 0 && R_obj.sensors.at(5) == 0 && R_obj.sensors.at(6) ==0 && R_obj.sensors.at(7) == 0) {
            if ((0<=R_obj.theta && 45>= R_obj.theta)||(315<R_obj.theta && 360>= R_obj.theta)) {
                if (VERBOSE) {
                    cout<<"Pass Quad 0"<<endl;
                }
                check_pass = true;
            }
            
        }else  if (R_obj.sensors.at(4) == 0 && R_obj.sensors.at(5) != 0 && R_obj.sensors.at(6) ==0 && R_obj.sensors.at(7) == 0) {
            if((45<R_obj.theta && 135>= R_obj.theta)){
                if (VERBOSE) {
                    cout<<"Pass Quad 1";
                }
                check_pass = true;
            }
        }else if (R_obj.sensors.at(4) == 0 && R_obj.sensors.at(5) == 0 && R_obj.sensors.at(6) !=0 && R_obj.sensors.at(7) == 0) {
            if((135<R_obj.theta && 225>= R_obj.theta)){
                if (VERBOSE) {
                    cout<<"Pass Quad 2";
                }
                check_pass = true;
            }
        }else if (R_obj.sensors.at(4) == 0 && R_obj.sensors.at(5) == 0 && R_obj.sensors.at(6) ==0 && R_obj.sensors.at(7) != 0) {
            if((225<R_obj.theta && 315>= R_obj.theta)){
                if (VERBOSE) {
                    cout<<"Pass Quad 3";
                }
                check_pass = true;
            }
        }else{
            cout<<"Issue at an angle ::"<<R_obj.theta<<" with x_position and y_position"<<P_obj.x_position_poi<<P_obj.y_position_poi<<endl;
            exit(10);
        }
        assert(check_pass==true);
        i+=7;
        R_obj.theta+=7;
        R_obj.reset_sensors();
    }
    
}

vector<double> row_values;
vector< vector <double> > assert_check_values;

void fill_assert_check_values(){
    //First set of x , y thetha values
    for(int i=0;i<3;i++)
        row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //second set of x,y,thetha values
    row_values.push_back(0);
    row_values.push_back(1);
    row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //third set of x,y,thetha values
    row_values.push_back(1);
    row_values.push_back(2);
    row_values.push_back(45);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //fourth set of x,y,thetha values
    row_values.push_back(1);
    row_values.push_back(3);
    row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //fifth set of x,y,thetha values
    row_values.push_back(0);
    row_values.push_back(4);
    row_values.push_back(315);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //sixth set of x,y,thetha values
    row_values.push_back(0);
    row_values.push_back(5);
    row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
}

bool tolerance(double delta_maniplate,double check_value){
    double delta = 0.0000001;
    if (((delta+ delta_maniplate)>check_value)|| ((delta- delta_maniplate)<check_value) || (( delta_maniplate)==check_value)) {
        return true;
    }else{
        return false;
    }
}


void test_path(double x_start, double y_start){
    bool VERBOSE = false;
    Rover R_obj;
    POI P_obj;
    
    //given
    R_obj.x_position=x_start;
    R_obj.y_position=y_start;
    R_obj.theta=0.0;
    
    P_obj.x_position_poi=1.0;
    P_obj.y_position_poi=1.0;
    P_obj.value_poi=100;
    
    
    
    fill_assert_check_values();
    
    int step_number = 0;
    bool check_assert = false;
    
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==0) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    double dx=0.0,dy=1.0;
    R_obj.move_rover(dx, dy);
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==1) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    
    dx=1.0;
    dy=1.0;
    R_obj.move_rover(dx, dy);
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==2) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    dx=-1/sqrt(2.0);
    dy=1/sqrt(2.0);
    R_obj.move_rover(dx, dy);
    R_obj.reset_sensors();
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==3) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    dx=-1.0;
    dy=1.0;
    R_obj.move_rover(dx, dy);
    R_obj.reset_sensors();
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==4) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    dx=1/sqrt(2.0);
    dy=1/sqrt(2.0);
    R_obj.move_rover(dx, dy);
    R_obj.reset_sensors();
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==5) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
}

vector< vector <double> > point_x_y_circle;
vector<double> temp;

void find_x_y_test_circle_path(double start_x_position,double start_y_position,double angle){
    double radius = 1.0;
    temp.push_back(start_x_position+(radius*cos(angle * (PI /180))));
    temp.push_back(start_y_position+(radius*sin(angle * (PI/180))));
}

void test_circle_path(double x_start,double y_start){
    bool VERBOSE = false;
    Rover R_obj;
    POI P_obj;
    
    P_obj.x_position_poi=0.0;
    P_obj.y_position_poi=0.0;
    P_obj.value_poi=100.0;
    
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    
    double dx=0.0,dy=1.0;
    double angle=0.0;
    
    for(;angle<=360;){
        R_obj.x_position=x_start;
        R_obj.y_position=y_start;
        R_obj.theta=0.0;
        find_x_y_test_circle_path(x_start, y_start,angle);
        dx=temp.at(0);
        dy=temp.at(1);
        R_obj.move_rover(dx, dy);
        assert(tolerance(R_obj.x_position, dx));
        assert(tolerance(R_obj.y_position, dy));
        assert(tolerance(R_obj.theta, angle));
        temp.clear();
        angle+=15.0;
    }
    
}

void test_all_sensors(){
    POI_sensor_test();
    rover_sensor_test();
    custom_test();
    double x_start = 0.0, y_start = 0.0;
    stationary_rover_test(x_start,y_start);
    stationary_rover_test_1(x_start, y_start);
    stationary_poi_test(x_start,y_start);
    two_rovers_test(x_start,y_start);
    test_path(x_start,y_start);
    x_start = 0.0, y_start = 0.0;
    test_circle_path(x_start,y_start);
}


/*************************************************************************
 Create Teams
 ***********************************************************************/

void create_teams(vector<Rover>* p_rover, int numNN){
    bool verbose = false;
    bool print_text = true;
    if (verbose) {
        cout<<"This are team numbers<<<"<<endl;
        for (int rover_number = 0; rover_number < p_rover->size(); rover_number++) {
            for (int policy = 0; policy < p_rover->at(rover_number).network_for_agent.size(); policy++) {
                cout<<p_rover->at(rover_number).network_for_agent.at(policy).my_team_number<<"\t";
            }
            cout<<endl;
        }
    }
    
    //Create teams
    for (int team_number = 0; team_number < numNN; team_number++) {
        vector<int> temp_number_holder;
        for (int rover_number = 0; rover_number < p_rover->size(); rover_number++) {
            int temp = rand()%numNN;
            for (int size_number = 0; size_number < temp_number_holder.size(); size_number++) {
                
                //                if (temp_number_holder.at(size_number) == temp) {
                //                    temp = rand()%numNN;
                //                    size_number = -1;
                //                }
                
                while (p_rover->at(rover_number).network_for_agent.at(temp).my_team_number != 9999999) {
                    temp = rand()%numNN;
                }
            }
            while (p_rover->at(rover_number).network_for_agent.at(temp).my_team_number != 9999999) {
                temp = rand()%numNN;
            }
            temp_number_holder.push_back(temp);
        }
        if (verbose) {
            for (int temp = 0; temp<temp_number_holder.size(); temp++) {
                cout<< temp_number_holder.at(temp)<<"\t";
            }
            cout<<endl;
        }
        
        assert(temp_number_holder.size() == p_rover->size());
        
        for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
            p_rover->at(rover_number).network_for_agent.at(temp_number_holder.at(rover_number)).my_team_number = team_number;
        }
        
        if (verbose) {
            cout<<"This are team numbers<<<"<<endl;
            for (int rover_number = 0; rover_number < p_rover->size(); rover_number++) {
                for (int policy = 0; policy < p_rover->at(rover_number).network_for_agent.size(); policy++) {
                    cout<<p_rover->at(rover_number).network_for_agent.at(policy).my_team_number<<"\t";
                }
                cout<<endl;
            }
        }
    }
    
    if (verbose) {
        for (int rover_number = 0; rover_number < p_rover->size(); rover_number++) {
            for (int nn = 0; nn < p_rover->at(rover_number).network_for_agent.size(); nn++) {
                cout<<p_rover->at(rover_number).network_for_agent.at(nn).my_team_number <<endl;
            }
        }
    }
    
    //make sure policy of same rover doesn't have same team number
    for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
        for (int nn = 0; nn<p_rover->at(rover_number).network_for_agent.size(); nn++) {
            for (int nn_1 = 0; nn_1 < p_rover->at(rover_number).network_for_agent.size(); nn_1++) {
                if (nn != nn_1) {
                    assert(p_rover->at(rover_number).network_for_agent.at(nn).my_team_number != 9999999);
                    assert(p_rover->at(rover_number).network_for_agent.at(nn).my_team_number != p_rover->at(rover_number).network_for_agent.at(nn_1).my_team_number);
                }
            }
        }
    }
    
    if (print_text) {
        FILE* p_print_text;
        p_print_text = fopen("/Users/adarshkesireddy/Desktop/File_analysis/Teams.txt", "a");
        for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
            for (int policy_number = 0 ; policy_number < p_rover->at(rover_number).network_for_agent.size(); policy_number++) {
                fprintf(p_print_text, "%d \t", p_rover->at(rover_number).network_for_agent.at(policy_number).my_team_number);
            }
            fprintf(p_print_text, "\n");
        }
        fclose(p_print_text);
    }
}

void set_teams_to_inital(vector<Rover>* p_rover, int numNN){
    //set all team numbers to 9999999
    for (int rover_number = 0; rover_number< p_rover->size(); rover_number++) {
        for (int nn = 0; nn < p_rover->at(rover_number).network_for_agent.size(); nn++) {
            p_rover->at(rover_number).network_for_agent.at(nn).my_team_number = 9999999;
        }
    }
}


/****************************************************************************
 Same old EA
 **************************************************************************/

void repopulate(vector<Rover>* teamRover,int number_of_neural_network){
    for (int rover_number =0; rover_number < teamRover->size(); rover_number++) {
        vector<unsigned> a;
        for (int neural_network =0; neural_network < (number_of_neural_network/2); neural_network++) {
            int R = rand()%teamRover->at(rover_number).network_for_agent.size();
            Net N(a);
            N = teamRover->at(rover_number).network_for_agent.at(R);
            N.mutate();
            teamRover->at(rover_number).network_for_agent.push_back(N);
        }
        assert(teamRover->at(rover_number).network_for_agent.size() == number_of_neural_network);
    }
}

/*
void repopulate(vector<Rover>* teamRover,int number_of_neural_network){
    for (int rover_number =0; rover_number < teamRover->size(); rover_number++) {
        //vector<unsigned> a;
        for (int neural_network =0; neural_network < (number_of_neural_network/2); neural_network++) {
            int R = rand()%teamRover->at(rover_number).network_for_agent.size();
            //Net N(a);
            //N = teamRover->at(rover_number).network_for_agent.at(R);
            //N.mutate();
            //teamRover->at(rover_number).network_for_agent.push_back(N);
            teamRover->at(rover_number).network_for_agent.push_back(teamRover->at(rover_number).network_for_agent.at(R));
            teamRover->at(rover_number).network_for_agent.back().mutate();
        }
        assert(teamRover->at(rover_number).network_for_agent.size() == number_of_neural_network);
    }
}
*/
void ccea(vector<Rover>* teamRover,POI* individualPOI, int numNN, int number_of_objectives,int reward_number){
    bool verbose = false;
    
    /*FILE* p_before;
    p_before = fopen("before.txt", "a");
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy_number = 0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
            fprintf(p_before, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team);
        }
        fprintf(p_before, "\n");
    }
    
    for (int rover = 0 ; rover < teamRover->size(); rover++) {
        for (int policy = 0 ; policy < teamRover->at(rover).network_for_agent.size(); policy++) {
            fprintf(p_before, "%d \t",teamRover->at(rover).network_for_agent.at(policy).my_team_number);
        }
        fprintf(p_before, "\n");
    }
    fclose(p_before);
    
    FILE* p_difference_data;
    p_difference_data = fopen("Difference_Data.txt", "a");
    */
    // Remove low fitness policies
    for (int rover_number = 0; rover_number < teamRover->size(); rover_number++) {
        if (verbose) {
            cout<<"Rover Number \t :::"<<rover_number<<endl;
            for (int prrint_rover_number = 0 ; prrint_rover_number < teamRover->size(); prrint_rover_number++) {
                for (int print_policy_number = 0 ; print_policy_number < teamRover->at(prrint_rover_number).network_for_agent.size(); print_policy_number++) {
                    cout<<teamRover->at(prrint_rover_number).network_for_agent.at(print_policy_number).global_reward_wrt_team <<"\t";
                }
                cout<<endl;
            }
        }
        
//        FILE* p_global;
//        p_global = fopen("Local.txt", "a");
//        fprintf(p_global, "\n");
//        for (int policy_number = 0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
//            fprintf(p_global, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy_number).local_reward_wrt_team);
//        }
        
        
        for (int policy = 0; policy < numNN/2; policy++) {
            if (verbose) {
                cout<<"policy \t :::"<<policy<<endl;
            }
            int random_number_1 = rand()%teamRover->at(rover_number).network_for_agent.size();
            int random_number_2 = rand()%teamRover->at(rover_number).network_for_agent.size();
            while ((random_number_1 == random_number_2) || (random_number_1 == teamRover->at(rover_number).network_for_agent.size()) || (random_number_2 == teamRover->at(rover_number).network_for_agent.size())) {
                random_number_2 = rand()%teamRover->at(rover_number).network_for_agent.size();
                random_number_1 = rand()%teamRover->at(rover_number).network_for_agent.size();
            }
            
            if (verbose) {
                cout<< random_number_1<<"\t"<<random_number_2<<endl;
            }
            
            //fprintf(p_difference_data, "%d \t %d \t", random_number_1,random_number_2);
            
            //Select 1 for local reward 2 for global reward 3 for difference reward
            
            double fitness_1 = 0.0;
            double fitness_2 = 0.0;
            
            int type_of_selection = reward_number;
            switch (type_of_selection) {
                case 1:
                    fitness_1 = teamRover->at(rover_number).network_for_agent.at(random_number_1).local_reward_wrt_team;
                    fitness_2 = teamRover->at(rover_number).network_for_agent.at(random_number_2).local_reward_wrt_team;
                    break;
                    
                case 2:
                    fitness_1 = teamRover->at(rover_number).network_for_agent.at(random_number_1).global_reward_wrt_team;
                    fitness_2 = teamRover->at(rover_number).network_for_agent.at(random_number_2).global_reward_wrt_team;
                    break;
                    
                case 3:
                    fitness_1 = teamRover->at(rover_number).network_for_agent.at(random_number_1).difference_reward_wrt_team;
                    fitness_2 = teamRover->at(rover_number).network_for_agent.at(random_number_2).difference_reward_wrt_team;
                    break;
                
                case 4:
                    fitness_1 = teamRover->at(rover_number).network_for_agent.at(random_number_1).global_reward_wrt_team - teamRover->at(rover_number).network_for_agent.at(random_number_1).difference_reward_wrt_team;
                    fitness_2 = teamRover->at(rover_number).network_for_agent.at(random_number_2).global_reward_wrt_team - teamRover->at(rover_number).network_for_agent.at(random_number_2).difference_reward_wrt_team;
                    break;
                    
                case 5:
                    
                    break;
            }
            
            if (fitness_1 > fitness_2) {
                teamRover->at(rover_number).network_for_agent.erase(teamRover->at(rover_number).network_for_agent.begin()+random_number_2);
            }else{
                teamRover->at(rover_number).network_for_agent.erase(teamRover->at(rover_number).network_for_agent.begin()+random_number_1);
            }
        }
    }
    //fclose(p_difference_data);
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        assert(teamRover->at(rover_number).network_for_agent.size() == numNN/2);
    }
    
    /*FILE* p_after;
    p_after = fopen("after.txt", "a");
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy_number = 0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
            fprintf(p_after, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team);
        }
        fprintf(p_after, "\n");
    }
    
    for (int rover = 0 ; rover < teamRover->size(); rover++) {
        for (int policy = 0 ; policy < teamRover->at(rover).network_for_agent.size(); policy++) {
            fprintf(p_before, "%d \t",teamRover->at(rover).network_for_agent.at(policy).my_team_number);
        }
        fprintf(p_before, "\n");
    }
    fclose(p_after);
    */
    //Repopulate other networks
    repopulate(teamRover, numNN);
    
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        assert(teamRover->at(rover_number).network_for_agent.size() == numNN);
    }
}


void set_rover_to_init(vector<Rover>* teamRover){
    for (int rover_number = 0; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).x_position = teamRover->at(rover_number).x_position_vec.at(0);
        teamRover->at(rover_number).y_position = teamRover->at(rover_number).y_position_vec.at(0);
        teamRover->at(rover_number).theta = 0.0;
    }
}

/********************************************************************************
 simulation: Following things happen here and flow is same
 1. policy numbers are saved into a vector
 2. setting rovers to initial position
 3. moving rovers in each time step and updating its distance to POI
 4. for each policy in team calculate local reward
 5. for each policy in team calculate global reward ( This will be same for each team)
 6. for each policu in team calculate difference reward
 ********************************************************************************/

void simulation_new_version( vector<Rover>* teamRover, POI* individualPOI,double scaling_number, int policy, int rover_number,int generation){
    bool full_verbose  = false;
    bool verbose = false;
    
    int local_policy = policy;
    int local_rover_number = rover_number;
    
    if (full_verbose) {
        cout<<"Locations of POI"<<endl;
        for (int temp_number = 0; temp_number < individualPOI->x_position_poi_vec.size(); temp_number++) {
            cout<<individualPOI->x_position_poi_vec.at(temp_number)<<"\t"<<individualPOI->y_position_poi_vec.at(temp_number)<<endl;
        }
    }
    
    //FILE* p_xy;
    //p_xy = fopen("XY.txt","a");
    
    //setting all rovers to inital state
    for (int temp_rover_number =0 ; temp_rover_number<teamRover->size(); temp_rover_number++) {
        teamRover->at(temp_rover_number).x_position = teamRover->at(temp_rover_number).x_position_vec.at(0);
        teamRover->at(temp_rover_number).y_position = teamRover->at(temp_rover_number).y_position_vec.at(0);
        teamRover->at(temp_rover_number).theta = 0.0;
    }
    
    for (int time_step = 0 ; time_step < 500 ; time_step++) {
        
        if (verbose || full_verbose) {
            cout<<"Print X and Y location"<<endl;
            cout<<teamRover->at(local_rover_number).x_position<<"\t"<<teamRover->at(local_rover_number).y_position<<endl;
        }
        //fprintf(p_xy, "%f \t %f \n",teamRover->at(local_rover_number).x_position,teamRover->at(local_rover_number).y_position);
        
        
        //reset_sense_new(rover_number, p_rover, p_poi); // reset and sense new values
        teamRover->at(local_rover_number).reset_sensors(); // Reset all sensors
        teamRover->at(local_rover_number).sense_all_values(individualPOI->x_position_poi_vec, individualPOI->y_position_poi_vec, individualPOI->value_poi_vec); // sense all values
        
        //Change of input values
        for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(local_rover_number).sensors.size(); change_sensor_values++) {
            teamRover->at(local_rover_number).sensors.at(change_sensor_values) /= scaling_number;
        }
        
        teamRover->at(local_rover_number).network_for_agent.at(local_policy).feedForward(teamRover->at(local_rover_number).sensors); // scaled input into neural network
        for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(local_rover_number).sensors.size(); change_sensor_values++) {
            assert(!isnan(teamRover->at(rover_number).sensors.at(change_sensor_values)));
        }
        
        double dx = teamRover->at(local_rover_number).network_for_agent.at(local_policy).outputvaluesNN.at(0);
        double dy = teamRover->at(local_rover_number).network_for_agent.at(local_policy).outputvaluesNN.at(1);
        teamRover->at(local_rover_number).network_for_agent.at(local_policy).outputvaluesNN.clear();
        
        assert(!isnan(dx));
        assert(!isnan(dy));
        teamRover->at(local_rover_number).move_rover(dx, dy);
        
        
        for (int cal_dis =0; cal_dis<individualPOI->value_poi_vec.size(); cal_dis++) {
            double x_distance_cal =((teamRover->at(local_rover_number).x_position) -(individualPOI->x_position_poi_vec.at(cal_dis)));
            double y_distance_cal = ((teamRover->at(local_rover_number).y_position) -(individualPOI->y_position_poi_vec.at(cal_dis)));
            double distance = sqrt((x_distance_cal*x_distance_cal)+(y_distance_cal*y_distance_cal));
            if (teamRover->at(local_rover_number).network_for_agent.at(local_policy).closest_dist_to_poi.at(cal_dis) > distance) {
                teamRover->at(local_rover_number).network_for_agent.at(local_policy).closest_dist_to_poi.at(cal_dis) = distance ;
            }
        }
        
        if (full_verbose) {
            cout<<"Print out Distances:: "<<endl;
            for (int temp_cal_distance = 0 ; temp_cal_distance < teamRover->at(local_rover_number).network_for_agent.at(local_policy).closest_dist_to_poi.size(); temp_cal_distance++) {
                cout<< teamRover->at(local_rover_number).network_for_agent.at(local_policy).closest_dist_to_poi.at(temp_cal_distance)<<endl;
            }
        }
    }
    
    //fclose(p_xy);
}

//Jan28 2018

void simulation_2(vector<Rover>* teamRover, POI* individualPOI, double scaling_number, int team_number){
    //Index_number contains index of each team in each rover
    vector<int> index_number;
    vector<int>* p_index_number = &index_number;
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            if (teamRover->at(rover_number).network_for_agent.at(policy).my_team_number == team_number) {
                index_number.push_back(policy);
            }
        }
    }
    assert(index_number.size() == teamRover->size());
    
    //Set everything to initial condition
    set_rover_to_init(teamRover);
    
    //Simulation
    for (int time_step = 0 ; time_step < 500; time_step++) {
        for (int current_rover = 0 ; current_rover < index_number.size(); current_rover++) {
            
            teamRover->at(current_rover).network_for_agent.at(index_number.at(current_rover)).store_x_values.push_back(teamRover->at(current_rover).x_position);
            teamRover->at(current_rover).network_for_agent.at(index_number.at(current_rover)).store_y_values.push_back(teamRover->at(current_rover).y_position);
            
            //reset sensors
            //reset_sense_new(rover_number, p_rover, p_poi); // reset and sense new values
            teamRover->at(current_rover).reset_sensors(); // Reset all sensors
            teamRover->at(current_rover).sense_values(individualPOI->x_position_poi_vec, individualPOI->y_position_poi_vec, individualPOI->value_poi_vec,p_index_number, current_rover,teamRover);// sense all values
            
            //Change of input values
            for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(current_rover).sensors.size(); change_sensor_values++) {
                teamRover->at(current_rover).sensors.at(change_sensor_values) /= scaling_number;
            }
            
            teamRover->at(current_rover).network_for_agent.at(index_number.at(current_rover)).feedForward(teamRover->at(current_rover).sensors); // scaled input into neural network
            for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(current_rover).sensors.size(); change_sensor_values++) {
                assert(!isnan(teamRover->at(current_rover).sensors.at(change_sensor_values)));
            }
            
            double dx = teamRover->at(current_rover).network_for_agent.at(index_number.at(current_rover)).outputvaluesNN.at(0);
            double dy = teamRover->at(current_rover).network_for_agent.at(index_number.at(current_rover)).outputvaluesNN.at(1);
            teamRover->at(current_rover).network_for_agent.at(index_number.at(current_rover)).outputvaluesNN.clear();
            
            assert(!isnan(dx));
            assert(!isnan(dy));
            teamRover->at(current_rover).move_rover(dx, dy);
            
            
            for (int cal_dis =0; cal_dis<individualPOI->value_poi_vec.size(); cal_dis++) {
                double x_distance_cal =((teamRover->at(current_rover).x_position) -(individualPOI->x_position_poi_vec.at(cal_dis)));
                double y_distance_cal = ((teamRover->at(current_rover).y_position) -(individualPOI->y_position_poi_vec.at(cal_dis)));
                double distance = sqrt((x_distance_cal*x_distance_cal)+(y_distance_cal*y_distance_cal));
                if (teamRover->at(current_rover).network_for_agent.at(index_number.at(current_rover)).closest_dist_to_poi.at(cal_dis) > distance) {
                    teamRover->at(current_rover).network_for_agent.at(index_number.at(current_rover)).closest_dist_to_poi.at(cal_dis) = distance ;
                }
            }
            
            assert(teamRover->at(current_rover).network_for_agent.at(index_number.at(current_rover)).closest_dist_to_poi.size() == individualPOI->value_poi_vec.size());
            
        }
    }
}

void calculate_rewards(vector<Rover>* teamRover,POI* individualPOI, int numNN, int number_of_objectives,vector<vector<int>>* p_poi_index){
    
    bool verbose = false;
    if (verbose) {
        //Making sure POI values are correct.
        for (int objective = 0 ; objective < number_of_objectives; objective++) {
            for (int poi = 0 ; poi < p_poi_index->at(objective).size(); poi++) {
                cout<<p_poi_index->at(objective).at(poi)<<"\t" << individualPOI->value_poi_vec.at(p_poi_index->at(objective).at(poi))<<"\n";
            }
            cout<<endl;
        }
    }
    
    //Closest distance calculation on 1
    for (int rover_number = 0; rover_number < teamRover->size(); rover_number++) {
        for (int policy_number = 0; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
            for (int closest_distance = 0; closest_distance < teamRover->at(rover_number).network_for_agent.at(policy_number).closest_dist_to_poi.size(); closest_distance++) {
                if (teamRover->at(rover_number).network_for_agent.at(policy_number).closest_dist_to_poi.at(closest_distance) < 1) {
                    teamRover->at(rover_number).network_for_agent.at(policy_number).closest_dist_to_poi.at(closest_distance) = 1;
                }
            }
        }
    }

    
    //Total Local reward
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            
            //make sure each POI has corresponding value assosicated to it
            assert(individualPOI->value_poi_vec.size() == teamRover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.size());
            
            //Actual Local Reward
            double temp_local_reward = 0;
            for (int closest_distance_number = 0; closest_distance_number<teamRover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.size(); closest_distance_number++) {
                temp_local_reward += ((individualPOI->value_poi_vec.at(closest_distance_number))/(teamRover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.at(closest_distance_number)));
            }
            teamRover->at(rover_number).network_for_agent.at(policy).local_reward_wrt_team = temp_local_reward;
        }
        
        
    }
    
    //Global reward
    //Find closest distance reached by team to POI
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy_number = 0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
            teamRover->at(rover_number).network_for_agent.at(policy_number).global_reward_wrt_team = -1;
        }
    }
    
    for (int team_number = 0 ; team_number < numNN; team_number++) {
        
        vector<int> team_index;
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                if (team_number == teamRover->at(rover_number).network_for_agent.at(policy).my_team_number) {
                    team_index.push_back(policy);
                }
            }
        }
        assert(team_index.size() == teamRover->size());
        
        vector<double> closest_distance;
        for (int poi_number =0 ; poi_number < individualPOI->value_poi_vec.size(); poi_number++) {
            double temp_distance  = 99999999.99999;
            for (int index = 0 ; index < team_index.size(); index++) {
                if (temp_distance > teamRover->at(index).network_for_agent.at(team_index.at(index)).closest_dist_to_poi.at(poi_number)) {
                    temp_distance = teamRover->at(index).network_for_agent.at(team_index.at(index)).closest_dist_to_poi.at(poi_number);
                }
            }
            closest_distance.push_back(temp_distance);
        }
        assert(closest_distance.size() == individualPOI->value_poi_vec.size());
        
        double temp_global = 0;
        for (int distance =0 ; distance < closest_distance.size(); distance++) {
            temp_global += (individualPOI->value_poi_vec.at(distance)/ closest_distance.at(distance));
        }
        
        //Combining both objectives
        for (int rover_number =0 ; rover_number <teamRover->size(); rover_number++) {
            for (int policy =0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                if (teamRover->at(rover_number).network_for_agent.at(policy).my_team_number == team_number) {
                    teamRover->at(rover_number).network_for_agent.at(policy).global_reward_wrt_team = temp_global;
                }
            }
        }
    }
    
    //Difference reward
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        
        for (int policy_number = 0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
            
            vector<double> difference_closest_distance;
            for (int cal_distance = 0 ; cal_distance < teamRover->at(rover_number).network_for_agent.at(policy_number).closest_dist_to_poi.size(); cal_distance++) {
                double temp_difference_distance = 999999999.99999;
                for (int temp_rover_number = 0 ; temp_rover_number < teamRover->size(); temp_rover_number++) {
                    if (rover_number != temp_rover_number) {
                        for (int temp_policy_number = 0 ; temp_policy_number<teamRover->at(temp_rover_number).network_for_agent.size() ; temp_policy_number++) {
                            if (teamRover->at(rover_number).network_for_agent.at(policy_number).my_team_number == teamRover->at(temp_rover_number).network_for_agent.at(temp_policy_number).my_team_number) {
                                //Then check for calculations
                                if (temp_difference_distance > teamRover->at(temp_rover_number).network_for_agent.at(temp_policy_number).closest_dist_to_poi.at(cal_distance)) {
                                    temp_difference_distance = teamRover->at(temp_rover_number).network_for_agent.at(temp_policy_number).closest_dist_to_poi.at(cal_distance);
                                }
                            }
                        }
                    }
                }
                difference_closest_distance.push_back(temp_difference_distance);
            }
            
            assert(difference_closest_distance.size() == individualPOI->value_poi_vec.size());
            
            //Calculate difference reward
            double temp_difference_reward = 0 ;
            for (int loop_counter = 0 ; loop_counter < difference_closest_distance.size(); loop_counter++) {
                temp_difference_reward += (individualPOI->value_poi_vec.at(loop_counter)/difference_closest_distance.at(loop_counter));
            }
            
            teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team = temp_difference_reward;
            
        }
    }
    
    bool print_reward = true;
    if (print_reward) {
        FILE* p_rewards;
        p_rewards = fopen("/Users/adarshkesireddy/Desktop/File_analysis/Rewards.txt", "a");
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int policy_number = 0 ; policy_number< teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
                fprintf(p_rewards, "%d \t %f \t %f \t %f \n",teamRover->at(rover_number).network_for_agent.at(policy_number).my_team_number,teamRover->at(rover_number).network_for_agent.at(policy_number).local_reward_wrt_team,teamRover->at(rover_number).network_for_agent.at(policy_number).global_reward_wrt_team,teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team);
            }
        }
        fprintf(p_rewards, "\n");
        fclose(p_rewards);
    }
    
    vector<vector<int>> team_index_numbers;
    vector<int> temp_index;
    for (int rover_number = 0; rover_number < 1; rover_number++) {
        for (int policy = 0 ; policy < teamRover->at(0).network_for_agent.size(); policy++) {
            temp_index.push_back(policy);
            for (int other_rover = 0 ; other_rover < teamRover->size(); other_rover++) {
                if (other_rover != rover_number) {
                    for (int other_policy = 0 ; other_policy < teamRover->at(other_rover).network_for_agent.size(); other_policy++) {
                        if (teamRover->at(other_rover).network_for_agent.at(other_policy).my_team_number == teamRover->at(rover_number).network_for_agent.at(policy).my_team_number) {
                            //Other policy index is what required
                            temp_index.push_back(other_policy);
                        }
                    }
                }
            }
            assert(temp_index.size() == teamRover->size());
            team_index_numbers.push_back(temp_index);
            temp_index.clear();
        }
    }
    assert(team_index_numbers.size() == numNN);
    
    //First find the number of points with same objective values and POI values
    vector<double> poi_values;
    for (int poi = 0 ; poi <individualPOI->value_poi_vec.size(); poi++) {
        poi_values.push_back(individualPOI->value_poi_vec.at(poi));
    }
    assert(poi_values.size() == individualPOI->value_poi_vec.size());
    
   
    vector<vector<int>> teams;
    vector<int> temp;
    for (int team_number = 0 ; team_number <numNN; team_number++) {
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                if (teamRover->at(rover_number).network_for_agent.at(policy).my_team_number == team_number) {
                    temp.push_back(policy);
                }
            }
        }
        teams.push_back(temp);
        temp.clear();
    }
    
    bool print_closest = true;
    if (print_closest) {
        FILE *p_close;
        p_close = fopen("/Users/adarshkesireddy/Desktop/File_analysis/Close.txt", "a");
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                for (int cl = 0 ; cl < teamRover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.size(); cl++) {
                    fprintf(p_close, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.at(cl));
                }
                fprintf(p_close, "\n");
            }
            fprintf(p_close, "\n");
        }
        fclose(p_close);
    }
    
    //Calculate
    // Run per objective. Then go for team number. For each team find out all the calculations like local, global and Difference
    //In this loop calculate local global difference with each objective
//    FILE* p_l;
//    p_l = fopen("ManualTest", "a");
    
    for (int team_number = 0 ; team_number < teams.size(); team_number++) {
//        fprintf(p_l, "Team Number \t :%d \n",team_number);
        // Calculating global reward
//        fprintf(p_l, "Global Reward \n");
        for (int objective = 0 ; objective < number_of_objectives; objective++) {
            double objective_value = 0.0;
            for (int poi = 0; poi < p_poi_index->at(objective).size(); poi++) {
                double closest_value = 0.0;
                for (int rover_number = 0 ; rover_number < teams.at(team_number).size(); rover_number++) {
                    //cout<<teams.at(team_number).at(rover_number)<<endl;
                    if (rover_number == 0) {
                        closest_value = teamRover->at(rover_number).network_for_agent.at(teams.at(team_number).at(rover_number)).closest_dist_to_poi.at(p_poi_index->at(objective).at(poi));
                    }
                    if (closest_value > teamRover->at(rover_number).network_for_agent.at(teams.at(team_number).at(rover_number)).closest_dist_to_poi.at(p_poi_index->at(objective).at(poi)) ) {
                        closest_value = teamRover->at(rover_number).network_for_agent.at(teams.at(team_number).at(rover_number)).closest_dist_to_poi.at(p_poi_index->at(objective).at(poi));
                    }
                }
                //cout<<individualPOI->value_poi_vec.at(poi)<<endl;
                //cout<<individualPOI->value_poi_vec.at(p_poi_index->at(objective).at(poi))<<endl;
                objective_value += (individualPOI->value_poi_vec.at(p_poi_index->at(objective).at(poi))/closest_value);
//                fprintf(p_l, "%f \n",closest_value);
            }
            for (int rover_number = 0 ; rover_number < teams.at(team_number).size(); rover_number++) {
                teamRover->at(rover_number).network_for_agent.at(teams.at(team_number).at(rover_number)).global_objective_values.push_back(objective_value);
            }
//            fprintf(p_l, "\n");
        }
        
        for (int rover_number = 0 ; rover_number < teams.at(team_number).size(); rover_number++) {
            assert(teamRover->at(rover_number).network_for_agent.at(teams.at(team_number).at(rover_number)).global_objective_values.size() == number_of_objectives);
        }
        
        //Calculating local reward
        for (int rover_number = 0 ; rover_number < teams.at(team_number).size(); rover_number++) {
            for (int objective = 0 ; objective < number_of_objectives; objective++) {
                double objective_value = 0.0;
                for (int poi = 0; poi < p_poi_index->at(objective).size(); poi++) {
                    objective_value += (individualPOI->value_poi_vec.at(p_poi_index->at(objective).at(poi))/teamRover->at(rover_number).network_for_agent.at(teams.at(team_number).at(rover_number)).closest_dist_to_poi.at(p_poi_index->at(objective).at(poi))) ;
                }
                teamRover->at(rover_number).network_for_agent.at(teams.at(team_number).at(rover_number)).local_objective_values.push_back(objective_value);
            }
        }
        
        for (int rover_number = 0 ; rover_number < teams.at(team_number).size(); rover_number++) {
            assert(teamRover->at(rover_number).network_for_agent.at(teams.at(team_number).at(rover_number)).global_objective_values.size() == number_of_objectives);
        }
        
        //Calculating difference reward
//        fprintf(p_l, "Difference Reward \n");
        for (int rover = 0 ; rover <teams.at(team_number).size(); rover++) {
            for (int objective = 0 ; objective < number_of_objectives; objective++) {
                double objective_v =0.0;
                for (int poi = 0 ; poi < p_poi_index->at(objective).size(); poi++) {
                    vector<double> distance_v;
                    for (int other_rover = 0 ; other_rover < teams.at(team_number).size(); other_rover++) {
                        if (rover != other_rover) {
                            distance_v.push_back(teamRover->at(other_rover).network_for_agent.at(teams.at(team_number).at(other_rover)).closest_dist_to_poi.at(p_poi_index->at(objective).at(poi)));
                        }
                    }
                    double dis;
                    for (int index = 0 ; index < distance_v.size(); index++) {
                        if (index == 0) {
                            dis = distance_v.at(index);
                        }
                        if (dis > distance_v.at(index)) {
                            dis = distance_v.at(index);
                        }
                    }
                    assert(distance_v.size() == teamRover->size()-1);
//                    for (int t = 0; t< distance_v.size(); t++) {
//                        fprintf(p_l, "%f \n",distance_v.at(t));
//                    }
                    objective_v += (individualPOI->value_poi_vec.at(p_poi_index->at(objective).at(poi))/dis);
                    distance_v.clear();
                }
                teamRover->at(rover).network_for_agent.at(teams.at(team_number).at(rover)).difference_objective_values.push_back(objective_v);
            }
        }
        
        for (int rover_number = 0 ; rover_number < teams.at(team_number).size(); rover_number++) {
            assert(teamRover->at(rover_number).network_for_agent.at(teams.at(team_number).at(rover_number)).difference_objective_values.size() == number_of_objectives);
        }
        
    }
//    fclose(p_l);
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            assert(teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.size() == teamRover->at(rover_number).network_for_agent.at(policy).local_objective_values.size());
            assert(teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.size() == teamRover->at(rover_number).network_for_agent.at(policy).global_objective_values.size());
        }
    }
    
    bool print_cal = true;
    
    if (print_cal) {
        FILE* p_cal;
        p_cal = fopen("/Users/adarshkesireddy/Desktop/File_analysis/Calculations.txt", "a");
        for (int rover_number =0 ; rover_number < teamRover->size(); rover_number++) {
            for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                for (int index = 0; index < teamRover->at(rover_number).network_for_agent.at(policy).global_objective_values.size(); index++) {
                    fprintf(p_cal,"%f \t %f \t %f \n", teamRover->at(rover_number).network_for_agent.at(policy).local_objective_values.at(index),teamRover->at(rover_number).network_for_agent.at(policy).global_objective_values.at(index),teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.at(index));
                }
                fprintf(p_cal, "\n");
            }
            fprintf(p_cal, "\n");
        }
        fclose(p_cal);
    }
}


/* Routine for usual non-domination checking
 It will return the following values
 1 if policy dominates other_policy
 -1 if other_policy dominates policy
 0 if both policy and other_policy are non-dominated */

int check_domination(int rover_number,int policy,int other_policy,vector<Rover>* teamRover){
    
    assert(teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.size() == teamRover->at(rover_number).network_for_agent.at(other_policy).difference_objective_values.size());
    
    bool verbose = false;
    if (verbose) {
        for (int objective = 0 ; objective < teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.size(); objective++) {
            cout<<teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.at(objective)<<"\t"<<teamRover->at(rover_number).network_for_agent.at(other_policy).difference_objective_values.at(objective)<<endl;
        }
    }
    
    //true means that value is better
    vector<bool> first_policy,second_policy;
    for (int i = 0 ; i< teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.size(); i++) {
        first_policy.push_back(false);
        second_policy.push_back(false);
    }
    
    for (int objective = 0 ; objective < teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.size(); objective++) {
        if (teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.at(objective) <0 && teamRover->at(rover_number).network_for_agent.at(other_policy).difference_objective_values.at(objective) < 0) {
            if (teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.at(objective) >teamRover->at(rover_number).network_for_agent.at(other_policy).difference_objective_values.at(objective)) {
                //policy is better
                first_policy.at(objective) = true;
            }else{
                if (teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.at(objective) < teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.at(objective)) {
                    //other policy is better
                    second_policy.at(objective) = true;
                }else{
                    //Both are non dominated
                }
            }
            
        }else{
            if (teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.at(objective) < 0 && teamRover->at(rover_number).network_for_agent.at(other_policy).difference_objective_values.at(objective) == 0) {
                //Other policy is better
                second_policy.at(objective) = true;
            }else if (teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.at(objective) == 0 && teamRover->at(rover_number).network_for_agent.at(other_policy).difference_objective_values.at(objective) < 0) {
                    //policy is better
                first_policy.at(objective) = true;
                }else{
                    //Here we are comparing both negatives
                    if (teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.at(objective) > teamRover->at(rover_number).network_for_agent.at(other_policy).difference_objective_values.at(objective)) {
                        //other policy is better
                        second_policy.at(objective) = true;
                    }else if (teamRover->at(rover_number).network_for_agent.at(policy).difference_objective_values.at(objective) < teamRover->at(rover_number).network_for_agent.at(other_policy).difference_objective_values.at(objective)) {
                        // policy is better
                        first_policy.at(objective) = true;
                    }else{
                        //Both are non dominated
                    }
                }
        }
    }
    bool first_all = true;
    bool second_all = true;
    for (int i = 0 ; i < first_policy.size(); i++) {
        if (first_policy.at(i) ) {
            first_all = false;
        }
        if (second_policy.at(i)) {
            second_all = false;
        }
    }
    
    if (first_all) {
        return 1;
    }else if (second_all){
        return -1;
    }
    
    return 0;
    
}

double calculate_distance(double x,double y, double x_1,double y_1){
    return sqrt(pow((x-x_1), 2)+pow((y-y_1), 2));
}


void fitness_assignment_fast_nondominated_sort(vector<Rover>* teamRover,int number_of_objectives){
    
    cout<< "In Fitness assignemnet function "<<endl;
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            teamRover->at(rover_number).network_for_agent.at(policy).front.clear();
            teamRover->at(rover_number).network_for_agent.at(policy).dominate.clear();
            teamRover->at(rover_number).network_for_agent.at(policy).num_donimated.clear();
        }
    }
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            teamRover->at(rover_number).network_for_agent.at(policy).front.push_back(vector<double>  (0));
            teamRover->at(rover_number).network_for_agent.at(policy).dominate.push_back(vector<double>  (0));
            teamRover->at(rover_number).network_for_agent.at(policy).num_donimated.push_back(0);
            teamRover->at(rover_number).network_for_agent.at(policy).dominating_me = 0;
            teamRover->at(rover_number).network_for_agent.at(policy).already_have_front = false;
        }
    }
    vector<vector<vector<int>>> finial_front;
    vector<vector<int>> front;
    vector<int> temp_front;
    bool next_fron = false;
    do {
        //bool next_fron = false;
        for (int rover_number  = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                for (int other_policy = 0 ; other_policy < teamRover->at(rover_number).network_for_agent.size(); other_policy++) {
                    if( (policy != other_policy)&&(teamRover->at(rover_number).network_for_agent.at(policy).already_have_front == false)) {
                        if (teamRover->at(rover_number).network_for_agent.at(other_policy).already_have_front == false) {
                            int temp = check_domination(rover_number, policy,other_policy,teamRover);
                            if (temp == 1) {
                                teamRover->at(rover_number).network_for_agent.at(policy).dominating_over.push_back(other_policy);
                            }else if (temp == -1){
                                teamRover->at(rover_number).network_for_agent.at(policy).dominating_me +=1;
                            }else{
                                //Next level
                                //next_fron = true;
                            }
                        }
                    }
                }
                if ((teamRover->at(rover_number).network_for_agent.at(policy).dominating_me == 0) && (teamRover->at(rover_number).network_for_agent.at(policy).already_have_front == false)) {
                    temp_front.push_back(rover_number);
                    temp_front.push_back(policy);
                    teamRover->at(rover_number).network_for_agent.at(policy).rank = 1;
                    teamRover->at(rover_number).network_for_agent.at(policy).already_have_front = true;
                    front.push_back(temp_front);
                    temp_front.clear();
                }
                teamRover->at(rover_number).network_for_agent.at(policy).dominating_me = 0;
            }
        }
        
        if (front.size()!= 0 ) {
            finial_front.push_back(front);
        }
        front.clear();
        
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                if (teamRover->at(rover_number).network_for_agent.at(policy).already_have_front == false) {
                    next_fron = true;
                    break;
                }else{
                    next_fron = false;
                }
            }
            if (next_fron) {
                break;
            }
        }
    } while (next_fron);
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            assert(teamRover->at(rover_number).network_for_agent.at(policy).already_have_front == true);
        }
    }
    
    FILE* p_front;
    p_front =fopen("/Users/adarshkesireddy/Desktop/File_analysis/Front.txt", "a");
    for (int i=0; i< finial_front.size(); i++) {
        for (int j= 0; j<finial_front.at(i).size(); j++) {
            for (int k = 0; k< finial_front.at(i).at(j).size(); k++) {
                fprintf(p_front, "%d \t",finial_front.at(i).at(j).at(k));
            }
            fprintf(p_front, "\n");
        }
        fprintf(p_front, "\n");
    }
    fclose(p_front);
    
    //Calculate distance between each point in same front
    for (int front_number = 0 ; front_number < finial_front.size(); front_number++) {
        for (int current_index = 0 ; current_index < finial_front.at(front_number).size(); current_index++) {
            for (int other_index = 0 ; other_index < finial_front.at(front_number).size(); other_index++) {
                if ((current_index != other_index)&&(finial_front.at(front_number).at(current_index).at(0) == finial_front.at(front_number).at(other_index).at(0))) {
                    //calculate the distance
                    int rover_number = finial_front.at(front_number).at(current_index).at(0);
                    int net_number = finial_front.at(front_number).at(current_index).at(1);
                    int other_number = finial_front.at(front_number).at(other_index).at(1);
                    
                    double distance = calculate_distance(teamRover->at(rover_number).network_for_agent.at(net_number).difference_objective_values.at(0), teamRover->at(rover_number).network_for_agent.at(net_number).difference_objective_values.at(1), teamRover->at(rover_number).network_for_agent.at(other_number).difference_objective_values.at(0), teamRover->at(rover_number).network_for_agent.at(other_number).difference_objective_values.at(1));
                    
                    teamRover->at(rover_number).network_for_agent.at(net_number).crowding_distance.push_back(distance);
                }
            }
        }
    }
    
    for (int front_number = 0 ; front_number < finial_front.size(); front_number++) {
        for (int current_index = 0; current_index < finial_front.at(front_number).size(); current_index++) {
            for (int other_index = 0 ; other_index < finial_front.at(front_number).size(); other_index++) {
                if ( (current_index!= other_index) && (finial_front.at(front_number).at(current_index).at(0) == finial_front.at(front_number).at(other_index).at(0)) ) {
                    //Temp values
                    
                }
            }
        }
    }
    
    FILE* p_crowding_distance;
    p_crowding_distance = fopen("/Users/adarshkesireddy/Desktop/File_analysis/Crowding_distance.txt", "a");
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            if (teamRover->at(rover_number).network_for_agent.at(policy).crowding_distance.size() != 0) {
                for (int index = 0 ; index < teamRover->at(rover_number).network_for_agent.at(policy).crowding_distance.size(); index++) {
                    fprintf(p_crowding_distance, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy).crowding_distance.at(index));
                }
                fprintf(p_crowding_distance, "\n");
            }else{
                fprintf(p_crowding_distance, "Null \n");
            }
        }
        fprintf(p_crowding_distance, "\n");
    }
    fclose(p_crowding_distance);
    
    
    
    
//    for (int front_number  = 0 ; front_number < finial_front.size(); front_number++) {
//        for (int current_index = 0 ; current_index < finial_front.at(front_number).size(); current_index++) {
//            int rover_number = finial_front.at(front_number).at(current_index).at(0);
//            vector<double> all_objective_distance;
//            for (int next_index = 0 ; next_index < finial_front.at(front_number).size(); next_index++) {
//                if ( (current_index != next_index) && (finial_front.at(front_number).at(next_index).at(0) == rover_number)) {
//                    for (int objective_number = 0 ; objective_number < teamRover->at(rover_number).network_for_agent.at(current_index).difference_objective_values.size(); objective_number++) {
//                        double objective_distance = teamRover->at(rover_number).network_for_agent.at(current_index).difference_objective_values.at(objective_number) - teamRover->at(rover_number).network_for_agent.at(next_index).difference_objective_values.at(objective_number);
//                        all_objective_distance.push_back(objective_distance);
//                    }
//                }
//            }
//        }
//    }
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            teamRover->at(rover_number).network_for_agent.at(policy).crowding_distance_rank = 0 ;
        }
    }
    
    for (int front_number = 0 ; front_number < finial_front.size(); front_number++) {
        
        vector<double> index_numbers;
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int current_index = 0 ; current_index < finial_front.at(front_number).size(); current_index++) {
                if (rover_number == finial_front.at(front_number).at(current_index).at(0)) {
                    index_numbers.push_back(finial_front.at(front_number).at(current_index).at(1));
                }
            }
            
            //collect all the values
            vector<vector<double>> objective_sort_values;
            for (int objective = 0 ; objective < number_of_objectives; objective++) {
                vector<double> temp_objective_sort_values;
                for (int i = 0 ; i< index_numbers.size(); i++) {
                    temp_objective_sort_values.push_back(teamRover->at(rover_number).network_for_agent.at(index_numbers.at(i)).difference_objective_values.at(objective));
                }
                objective_sort_values.push_back(temp_objective_sort_values);
                temp_objective_sort_values.clear();
            }
            assert(objective_sort_values.size() == number_of_objectives);
            
            //Sort them
            for (int size = 0; size < objective_sort_values.size(); size++) {
                sort(objective_sort_values.at(size).begin(),objective_sort_values.at(size).end());
            }
            
            //Save the indexes
            vector<vector<int>> sorted_indexes;
            for (int objective = 0 ; objective < objective_sort_values.size(); objective++) {
                vector<int> temp_sorted;
                for (int value = 0 ; value < objective_sort_values.at(objective).size(); value++) {
                    //now check if the value matches with networks value in rover
                    for (int i = 0 ; i<index_numbers.size(); i++) {
                        if (objective_sort_values.at(objective).at(value) == teamRover->at(rover_number).network_for_agent.at(index_numbers.at(i)).difference_objective_values.at(objective)) {
                            temp_sorted.push_back(index_numbers.at(i));
                        }
                    }
                }
                sorted_indexes.push_back(temp_sorted);
                temp_sorted.clear();
            }
 
            
            
            
            
            //clear
            objective_sort_values.clear();
            sorted_indexes.clear();
            
        }
    }
    
}


void nsga_ii(vector<Rover>* teamRover,int number_of_objectives){
    fitness_assignment_fast_nondominated_sort(teamRover,number_of_objectives); // front is created
}

void hof(vector<Rover>* teamRover,int number_of_objectives){
    
}


void perform_mo(vector<Rover>* teamRover,int number_of_objectives){
    //case_number
    //1 NSGA-II
    //2 HOF
    int case_number = 1;
    switch (case_number) {
        case 1:
            nsga_ii(teamRover,number_of_objectives);
            break;
        case 2:
            hof(teamRover,number_of_objectives);
            break;
            
        default:
            break;
    }
}


/***************************
 Main
 **************************/

int main(int argc, const char * argv[]) {
    cout << "Hello, World!\n"<<endl;
    remove("/Users/adarshkesireddy/Desktop/File_analysis/Calculations.txt");
    remove("/Users/adarshkesireddy/Desktop/File_analysis/Close.txt");
    remove("/Users/adarshkesireddy/Desktop/File_analysis/Crowding_distance.txt");
    remove("/Users/adarshkesireddy/Desktop/File_analysis/Front.txt");
    remove("/Users/adarshkesireddy/Desktop/File_analysis/Paths.txt");
    remove("/Users/adarshkesireddy/Desktop/File_analysis/POI_Location.txt");
    remove("/Users/adarshkesireddy/Desktop/File_analysis/Rewards.txt");
    remove("/Users/adarshkesireddy/Desktop/File_analysis/Teams.txt");
    
    bool print_text = false;
    srand((unsigned)time(NULL));
    if (test_simulation) {
        for(int i=0;i <100;i++){
        test_all_sensors();
        }
        cout<<"All Test"<<endl;
    }
    
    if (run_simulation) {
        
        //First set up environment
        int number_of_rovers = 2;
        int number_of_poi = number_of_rovers*5;
        int number_of_objectives = 2;
        
        //Set values of poi's
        POI individualPOI;
        POI* p_poi = &individualPOI;
        
        vector<int> temp_index;
        vector<int>* p_temp = &temp_index;
        vector<vector<int>> poi_index;
        vector<vector<int>>* p_poi_index = &poi_index;
        for (int poi_location = 0; poi_location < number_of_poi; poi_location++) {
            double rand_1 = random_global(100);
            double rand_2 = random_global(100);
            
            individualPOI.x_position_poi_vec.push_back(rand_1);
            individualPOI.y_position_poi_vec.push_back(rand_2);
            if (poi_location < number_of_poi/2) {
                individualPOI.value_poi_vec.push_back(100.0);
                temp_index.push_back(poi_location);
            }else{
                individualPOI.value_poi_vec.push_back(50.0);
                temp_index.push_back(poi_location);
            }
            
            if (poi_location == ((number_of_poi/2)-1)) {
                p_poi_index->push_back(temp_index);
                temp_index.clear();
            }
            if(poi_location == (number_of_poi - 1) ){
                p_poi_index->push_back(temp_index);
                temp_index.clear();
            }
        }
        
        
        //check if environment along with rovers are set properly
        assert(individualPOI.x_position_poi_vec.size() == individualPOI.y_position_poi_vec.size());
        assert(individualPOI.value_poi_vec.size() == individualPOI.y_position_poi_vec.size());
        assert(individualPOI.value_poi_vec.size() == number_of_poi);
        
        //vectors of rovers
        vector<Rover> teamRover;
        vector<Rover>* p_rover = &teamRover;
        Rover a;
        for (int i=0; i<number_of_rovers; i++) {
            teamRover.push_back(a);
        }
        
        for (int i=0 ; i<number_of_rovers; i++) {
            teamRover.at(i).x_position_vec.push_back(0+(0.5*i));
            teamRover.at(i).y_position_vec.push_back(0);
        }
        
        set_rover_to_init(p_rover);
        
        assert(teamRover.size() == number_of_rovers);
        
        bool print_locations = true;
        if (print_locations) {
            FILE* p_print_poi;
            p_print_poi = fopen("/Users/adarshkesireddy/Desktop/File_analysis/POI_Location.txt", "a");
            for (int poi_location = 0 ; poi_location < number_of_poi; poi_location++) {
                fprintf(p_print_poi, "%f \t %f \t %f\n", individualPOI.x_position_poi_vec.at(poi_location),individualPOI.y_position_poi_vec.at(poi_location),individualPOI.value_poi_vec.at(poi_location));
            }
            fprintf(p_print_poi, "\n");
            for (int rover_number = 0 ; rover_number < teamRover.size(); rover_number++) {
                fprintf(p_print_poi, "%f \t %f \n",teamRover.at(rover_number).x_position,teamRover.at(rover_number).y_position);
            }
            fclose(p_print_poi);
        }
        
        //Second set up neural networks
        //Create numNN of neural network with pointer
        int numNN = 10;
        vector<unsigned> topology;
        topology.clear();
        topology.push_back(8);
        topology.push_back(14);
        topology.push_back(2);
        
        for (int rover_number =0 ; rover_number < number_of_rovers; rover_number++) {
            teamRover.at(rover_number).create_neural_network_population(numNN, topology);
        }
        
        
        double scaling_number = find_scaling_number(p_rover,p_poi);
        
        for (int rover_number =0 ; rover_number < teamRover.size(); rover_number++) {
            for (int policy_number = 0; policy_number < teamRover.at(rover_number).network_for_agent.size(); policy_number++) {
                teamRover.at(rover_number).network_for_agent.at(policy_number).best_value_so_far = 0.0;
            }
        }
        
        
        //Generations
        for(int generation =0 ; generation < 1 ;generation++){
            
                //First Create teams
                set_teams_to_inital(p_rover, numNN);
                create_teams(p_rover, numNN);
            
            
                //Setting distance to largest
                for (int network_number =0 ; network_number <numNN; network_number++) {
                    for (int rover_number =0; rover_number < number_of_rovers; rover_number++) {
                        teamRover.at(rover_number).network_for_agent.at(network_number).closest_dist_to_poi.clear();
                        for (int poi_number =0; poi_number<number_of_poi; poi_number++) {
                            teamRover.at(rover_number).network_for_agent.at(network_number).closest_dist_to_poi.push_back(99999999.9999);
                        }
                        
                    }
                }
            
                for (int rover_number = 0 ; rover_number < teamRover.size(); rover_number++) {
                    for (int policy_number = 0 ; policy_number < teamRover.at(rover_number).network_for_agent.size(); policy_number++) {
                        assert(teamRover.at(rover_number).network_for_agent.at(policy_number).closest_dist_to_poi.size() == individualPOI.value_poi_vec.size());
                    }
                }
            
            int case_number = 2;
            
            switch (case_number) {
                case 1:
                    break;
                
                case 2:
                    for (int team_number =0  ; team_number < numNN; team_number++) {
                        simulation_2(p_rover, p_poi,scaling_number, team_number);
                    }
                    break;
                default:
                    break;
            }
            
            bool print_paths = true;
            if (print_paths) {
                FILE* p_paths;
                p_paths = fopen("/Users/adarshkesireddy/Desktop/File_analysis/Paths.txt", "a");
                for (int rover_number = 0 ; rover_number < teamRover.size(); rover_number++) {
                    for (int policy_number = 0 ; policy_number < teamRover.at(rover_number).network_for_agent.size(); policy_number++) {
                        for (int index = 0 ; index < teamRover.at(rover_number).network_for_agent.at(policy_number).store_x_values.size(); index++) {
                            fprintf(p_paths, "%f \t %f \n", teamRover.at(rover_number).network_for_agent.at(policy_number).store_x_values.at(index),teamRover.at(rover_number).network_for_agent.at(policy_number).store_y_values.at(index));
                        }
                        fprintf(p_paths,"\n");
                    }
                    fprintf(p_paths, "\n");
                }
            }
            
            calculate_rewards(p_rover,p_poi,numNN,number_of_objectives,p_poi_index);
            perform_mo(p_rover,number_of_objectives);
            
        }
    }
    return 0;
}
