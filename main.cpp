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
    
    //For team
    
    int my_team_number;
    int previous_team_number;
    
    //For team
    double local_reward_wrt_team;
    double global_reward_wrt_team;
    double difference_reward_wrt_team;
    double difference_reward_new;
    
    //Hall of fame
    bool hall_of_fame = false;
    vector<double> objective_reward_local;
    vector<double> objective_reward_global;
    vector<double> objective_reward_difference;
    
    //store X and Y position
    vector<double> temp_x;
    vector<double> temp_y;
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

void Rover::sense_poi(double poix, double poiy, double val){
    double delta = sense_poi_delta(poix, poiy);
    int quad = find_quad(poix,poiy);
    sensors.at(quad) += val/delta;
}

void Rover::sense_rover(double otherx, double othery){
    double delta = sense_poi_delta(otherx,othery);
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
    bool VERBOSE = false;
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


double find_scaling_number(){
    double number =0.0;
    double temp_number =0.0;
    vector<double> xposition;
    vector<double> yposition;
    Rover R_obj; //Rover object
    POI P_obj; // POI object
    
    P_obj.x_position_poi=50.0;
    P_obj.y_position_poi=100.0;
    P_obj.value_poi =100;
    
    int temp_rand = rand()%100;
    while (temp_rand == 0) {
        temp_rand = rand()%100;
    }
    vector < vector <double> > group_sensors;
    
    for (int temp=0; temp<temp_rand; temp++) {
        R_obj.x_position=rand()%100;
        R_obj.y_position=rand()%100;
        xposition.push_back(R_obj.x_position);
        yposition.push_back(R_obj.y_position);
        
        
        R_obj.reset_sensors();
        R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
        group_sensors.push_back(R_obj.sensors);
    }
    
    assert(!group_sensors.empty());
    
    for (int i=0; i<group_sensors.size(); i++) {
        temp_number=*max_element(group_sensors.at(i).begin(), group_sensors.at(i).end());
        if (temp_number>number) {
            number=temp_number;
        }
    }
    
    R_obj.reset_sensors();
    
    assert(number != 0.0);
    xposition.clear();
    yposition.clear();
    return number;
}


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
    bool print_text = false;
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
    
    for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
        for (int policy_number = 0 ; policy_number < p_rover->at(rover_number).network_for_agent.size(); policy_number++) {
            p_rover->at(rover_number).network_for_agent.at(policy_number).previous_team_number = p_rover->at(rover_number).network_for_agent.at(policy_number).my_team_number;
        }
    }
    
    if (print_text) {
        FILE* p_print_text;
        p_print_text = fopen("Teams", "a");
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

//void repopulate(vector<Rover>* teamRover,int number_of_neural_network){
//    for (int rover_number =0; rover_number < teamRover->size(); rover_number++) {
//        vector<unsigned> a;
//        for (int neural_network =0; neural_network < (number_of_neural_network/2); neural_network++) {
//            int R = rand()%teamRover->at(rover_number).network_for_agent.size();
//            Net N(a);
//            N = teamRover->at(rover_number).network_for_agent.at(R);
//            N.mutate();
//            teamRover->at(rover_number).network_for_agent.push_back(N);
//        }
//        assert(teamRover->at(rover_number).network_for_agent.size() == number_of_neural_network);
//    }
//}

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

void ccea(vector<Rover>* teamRover,POI* individualPOI, int numNN, int number_of_objectives){
    bool verbose = false;
    
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
            
            //Select 1 for local reward 2 for global reward 3 for difference reward
            
            int type_of_selection = 2;
            switch (type_of_selection) {
                case 1:
                    if (teamRover->at(rover_number).network_for_agent.at(random_number_1).local_reward_wrt_team > teamRover->at(rover_number).network_for_agent.at(random_number_2).local_reward_wrt_team) {
                        //kill two
                        teamRover->at(rover_number).network_for_agent.erase(teamRover->at(rover_number).network_for_agent.begin()+random_number_2);
                    }else{
                        //kill one
                        teamRover->at(rover_number).network_for_agent.erase(teamRover->at(rover_number).network_for_agent.begin()+random_number_1);
                    }
                    break;
                    
                case 2:
                    if (teamRover->at(rover_number).network_for_agent.at(random_number_1).global_reward_wrt_team > teamRover->at(rover_number).network_for_agent.at(random_number_2).global_reward_wrt_team) {
                        //kill two
                        teamRover->at(rover_number).network_for_agent.erase(teamRover->at(rover_number).network_for_agent.begin()+random_number_2);
                    }else{
                        //kill one
                        teamRover->at(rover_number).network_for_agent.erase(teamRover->at(rover_number).network_for_agent.begin()+random_number_1);
                    }
                    break;
                    
                case 3:
                    if (teamRover->at(rover_number).network_for_agent.at(random_number_1).difference_reward_wrt_team < teamRover->at(rover_number).network_for_agent.at(random_number_2).difference_reward_wrt_team) {
                        //kill two
                        teamRover->at(rover_number).network_for_agent.erase(teamRover->at(rover_number).network_for_agent.begin()+random_number_2);
                    }else{
                        //kill one
                        teamRover->at(rover_number).network_for_agent.erase(teamRover->at(rover_number).network_for_agent.begin()+random_number_1);
                    }
                    break;
                case 4:
                    if (teamRover->at(rover_number).network_for_agent.at(random_number_1).difference_reward_wrt_team > teamRover->at(rover_number).network_for_agent.at(random_number_2).difference_reward_wrt_team) {
                        //kill two
                        teamRover->at(rover_number).network_for_agent.erase(teamRover->at(rover_number).network_for_agent.begin()+random_number_2);
                    }else{
                        //kill one
                        teamRover->at(rover_number).network_for_agent.erase(teamRover->at(rover_number).network_for_agent.begin()+random_number_1);
                    }
                    break;
            }
            
            if (verbose) {
                for (int prrint_rover_number = 0 ; prrint_rover_number < teamRover->size(); prrint_rover_number++) {
                    for (int print_policy_number = 0 ; print_policy_number < teamRover->at(prrint_rover_number).network_for_agent.size(); print_policy_number++) {
                        cout<<teamRover->at(prrint_rover_number).network_for_agent.at(print_policy_number).difference_reward_wrt_team<<"\t";
                    }
                    cout<<endl;
                }
            }
        }
    }
    
    //    FILE* p_wts;
    //    p_wts = fopen("weights", "a");
    //    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
    //        for (int policy_number = 0; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
    //            for (int layer = 0; layer < teamRover->at(rover_number).network_for_agent.at(policy_number).z_layer.size(); layer++) {
    //                for (int neuron = 0 ; neuron < teamRover->at(rover_number).network_for_agent.at(policy_number).z_layer.at(layer).size(); neuron++) {
    //                    for (int  out_put_weights = 0; out_put_weights < teamRover->at(rover_number).network_for_agent.at(policy_number).z_layer.at(layer).at(out_put_weights).z_outputWeights.size(); out_put_weights++) {
    //                        fprintf(p_wts, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy_number).z_layer.at(layer).at(out_put_weights).z_outputWeights.at(out_put_weights).weight );
    //                    }
    //                }
    //                fprintf(p_wts, "\n");
    //            }
    //            fprintf(p_wts, "\n");
    //        }
    //        fprintf(p_wts, "\n");
    //    }
    //    fclose(p_wts);
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        assert(teamRover->at(rover_number).network_for_agent.size() == numNN/2);
    }
    
//    double check_difference_reward = 0.0;
//    vector<double> index;
//    for (int rover_number = 0; rover_number < teamRover->size(); rover_number++) {
//        for (int policy_number = 0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
//            if (check_difference_reward < teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team) {
//                check_difference_reward = teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team;
//            }
//        }
//    }
//    
//    for (int rover_number = 0; rover_number < teamRover->size(); rover_number++) {
//        for (int policy_number = 0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
//            if (teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team == check_difference_reward) {
//                index.push_back(rover_number);
//                index.push_back(policy_number);
//            }
//        }
//    }
//    vector<int> numbers_neuron;
//    numbers_neuron.push_back(9);
//    numbers_neuron.push_back(11);
//    numbers_neuron.push_back(3);
    
    //    for (int layers = 0; layers < teamRover->at(index.at(0)).network_for_agent.at(index.at(1)).z_layer.size(); layers++) {
    //        for (int loop_1 = 0; loop_1 < numbers_neuron.size(); loop_1++) {
    //            for (int loop_2 = 0; loop_2 < teamRover->at(index.at(0)).network_for_agent.at(index.at(1)).z_layer.at(layers).at(loop_1).z_outputWeights.size(); loop_2++) {
    //                cout<<teamRover->at(index.at(0)).network_for_agent.at(index.at(1)).z_layer.at(layers).at(loop_1).z_outputWeights.at(loop_2).weight<<endl;
    //            }
    //        }
    //
    //    }
    
    //Fill in the blank once
        repopulate(teamRover, numNN);
    
//    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
//        vector<unsigned> a;
//        for (int policy_number = 0; policy_number < (numNN/2); policy_number++) {
//            Net N(a);
//            int R = rand()%teamRover->at(rover_number).network_for_agent.size();
//            N = teamRover->at(rover_number).network_for_agent.at(R);
//            N.mutate();
//            teamRover->at(rover_number).network_for_agent.push_back(N);
//        }
//    }
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        assert(teamRover->at(rover_number).network_for_agent.size() == numNN);
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

void simulation_new_version( vector<Rover>* teamRover, POI* individualPOI,double scaling_number, int policy, int rover_number){
    bool full_verbose  = false;
    bool verbose = false;
    bool print_text = false;
    
    int local_policy = policy;
    int local_rover_number = rover_number;
    
    if (full_verbose) {
        cout<<"Locations of POI"<<endl;
        for (int temp_number = 0; temp_number < individualPOI->x_position_poi_vec.size(); temp_number++) {
            cout<<individualPOI->x_position_poi_vec.at(temp_number)<<"\t"<<individualPOI->y_position_poi_vec.at(temp_number)<<endl;
        }
    }
    
    //setting all rovers to inital state
    for (int temp_rover_number =0 ; temp_rover_number<teamRover->size(); temp_rover_number++) {
        teamRover->at(temp_rover_number).x_position = teamRover->at(temp_rover_number).x_position_vec.at(0);
        teamRover->at(temp_rover_number).y_position = teamRover->at(temp_rover_number).y_position_vec.at(0);
        teamRover->at(temp_rover_number).theta = 0.0;
    }
    
//    FILE* p_temp_text;
//    p_temp_text = fopen("X and Y coordinates", "a");
    
    
    for (int time_step = 0 ; time_step < 50000 ; time_step++) {
        
        if (verbose || full_verbose) {
            cout<<"Print X and Y location"<<endl;
            cout<<teamRover->at(local_rover_number).x_position<<"\t"<<teamRover->at(local_rover_number).y_position<<endl;
        }
        
//        fprintf(p_temp_text,"%f \t %f \n", teamRover->at(local_rover_number).x_position, teamRover->at(local_rover_number).y_position);
        
        
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
        
        teamRover->at(rover_number).network_for_agent.at(policy).temp_x.push_back(teamRover->at(rover_number).x_position);
        teamRover->at(rover_number).network_for_agent.at(policy).temp_y.push_back(teamRover->at(rover_number).y_position);
                                                                                  
        if (full_verbose) {
            cout<<"Print out Distances:: "<<endl;
            for (int temp_cal_distance = 0 ; temp_cal_distance < teamRover->at(local_rover_number).network_for_agent.at(local_policy).closest_dist_to_poi.size(); temp_cal_distance++) {
                cout<< teamRover->at(local_rover_number).network_for_agent.at(local_policy).closest_dist_to_poi.at(temp_cal_distance)<<endl;
            }
        }
    }
    
    
//    fclose(p_temp_text);
}

void calculate_rewards(vector<Rover>* teamRover,POI* individualPOI, int numNN, int number_of_objectives){
    bool full_verbose = false;
    bool verbose = false;
    
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
    
    if (full_verbose) {
        cout<<"Print"<<endl;
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                for (int distance = 0 ; distance < teamRover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.size(); distance++) {
                    cout<< teamRover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.at(distance)<<"\t";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        
        for (int rover_number = 0 ; rover_number< teamRover->size(); rover_number++) {
            for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                cout<<teamRover->at(rover_number).network_for_agent.at(policy).objective_reward_local.size()<<"\t";
            }
            cout<<endl;
        }
        
        for (int value = 0 ; value < individualPOI->value_poi_vec.size(); value++) {
            cout<< individualPOI->value_poi_vec.at(value)<<"\t";
        }
        cout<<endl;
        
    }
    
    
    
    //Objective Local Reward
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        //        cout<<rover_number<<endl;
        for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            //            cout<<policy<<endl;
            //            cout<<"Size of objective:: \t"<<teamRover->at(rover_number).network_for_agent.at(policy).objective_reward_local.size()<<endl;
            double value_check = -11111111.1111111;
            double temp_distance = 0 ;
            for (int distance = 0 ; distance < teamRover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.size(); distance++) {
                //                cout<<distance<<endl;
                //First time or when value of POI changes
                if ((distance == 0) || (value_check != individualPOI->value_poi_vec.at(distance))) {
                    if( (distance !=0)) {
                        //                        cout<<"Inside Push"<<endl;
                        teamRover->at(rover_number).network_for_agent.at(policy).objective_reward_local.push_back(temp_distance);
                    }
                    temp_distance = 0;
                    value_check = individualPOI->value_poi_vec.at(distance);
                    
                }
                temp_distance += ((individualPOI->value_poi_vec.at(distance))/(teamRover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.at(distance)));
            }
            teamRover->at(rover_number).network_for_agent.at(policy).objective_reward_local.push_back(temp_distance);
            assert(teamRover->at(rover_number).network_for_agent.at(policy).objective_reward_local.size() == number_of_objectives);
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
        
        //Each objective value
        double value_check = -11111.111;
        double temp_distance = 0 ;
        vector<double> temp_global_values;
        for (int distance = 0; distance < closest_distance.size(); distance++) {
            if ((distance == 0) || (value_check != individualPOI->value_poi_vec.at(distance))) {
                if( (distance !=0)) {
                    //                        cout<<"Inside Push"<<endl;
                    temp_global_values.push_back(temp_distance);
                }
                temp_distance = 0;
                value_check = individualPOI->value_poi_vec.at(distance);
            }
            temp_distance += ((individualPOI->value_poi_vec.at(distance))/(closest_distance.at(distance)));
        }
        temp_global_values.push_back(temp_distance);
        assert(temp_global_values.size() == number_of_objectives);
        
        
        for (int rover_number = 0; rover_number <teamRover->size(); rover_number++) {
            for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                if (teamRover->at(rover_number).network_for_agent.at(policy).my_team_number == team_number) {
                    for (int push_values = 0 ; push_values < temp_global_values.size(); push_values++) {
                        teamRover->at(rover_number).network_for_agent.at(policy).objective_reward_global.push_back(temp_global_values.at(push_values));
                    }
                }
            }
        }
        
    }
    
    for (int rover_number =0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            assert(teamRover->at(rover_number).network_for_agent.at(policy).objective_reward_global.size() == number_of_objectives);
        }
    }
    
    
    //Calculate difference reward
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        if (verbose) {
            cout<<"Rover ::"<<rover_number<<endl;
        }
        for (int policy_number = 0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
            if (verbose) {
                cout<<"Policies ::"<<policy_number<<endl;
            }
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
            if (full_verbose) {
                for (int loop_counter = 0 ; loop_counter < difference_closest_distance.size(); loop_counter++) {
                    cout<<difference_closest_distance.at(loop_counter)<<"\t";
                }
                cout<<endl;
            }
            assert(difference_closest_distance.size() == individualPOI->value_poi_vec.size());
            
            //Calculate difference reward
            double temp_difference_reward = 0 ;
            for (int loop_counter = 0 ; loop_counter < difference_closest_distance.size(); loop_counter++) {
                temp_difference_reward += ((individualPOI->value_poi_vec.at(loop_counter))/(difference_closest_distance.at(loop_counter)));
            }
            if (full_verbose) {
                cout<<"This is temp_difference_reward::::"<<temp_difference_reward<<endl;
            }
            
            teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team = temp_difference_reward;
            
            //            for (int other_rover = 0 ; other_rover < teamRover->size(); other_rover++) {
            ////                cout<<"Inside loop rover number ::::::::::: \t"<<other_rover<<endl;
            //                if ((other_rover != rover_number)) {
            //                    for (int other_policy = 0 ; other_policy < teamRover->at(other_rover).network_for_agent.size(); other_policy++) {
            //                        if (teamRover->at(other_rover).network_for_agent.at(other_policy).my_team_number == teamRover->at(rover_number).network_for_agent.at(policy_number).my_team_number) {
            ////                            cout<<"Are you coming here"<<endl;
            //                            teamRover->at(other_rover).network_for_agent.at(other_policy).difference_reward_wrt_team = temp_difference_reward;
            //                        }
            //                    }
            //                }
            //            }
            
            //Calculate objective reward
            double value_check = -11111.111;
            double temp_distance = 0 ;
            vector<double> temp_difference_values;
            for (int distance = 0; distance < difference_closest_distance.size(); distance++) {
                if ((distance == 0) || (value_check != individualPOI->value_poi_vec.at(distance))) {
                    if( (distance !=0)) {
                        //                        cout<<"Inside Push"<<endl;
                        temp_difference_values.push_back(temp_distance);
                    }
                    temp_distance = 0;
                    value_check = individualPOI->value_poi_vec.at(distance);
                }
                temp_distance += ((individualPOI->value_poi_vec.at(distance))/(difference_closest_distance.at(distance)));
            }
            temp_difference_values.push_back(temp_distance);
            assert(temp_difference_values.size() == number_of_objectives);
            
            for (int loop_counter = 0 ; loop_counter < temp_difference_values.size(); loop_counter++) {
                teamRover->at(rover_number).network_for_agent.at(policy_number).objective_reward_difference.push_back(temp_difference_values.at(loop_counter));
            }
            
        }
    }
    
    //    for (int rover_number = 0; rover_number < teamRover->size(); rover_number++) {
    //        for (int policy_number = 0 ; policy_number < teamRover->at(rover_number).network_for_agent.size() ; policy_number++) {
    //            teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team = teamRover->at(rover_number).network_for_agent.at(policy_number).global_reward_wrt_team - teamRover->at(rover_number).network_for_agent.at(policy_number).local_reward_wrt_team;
    //        }
    //    }
    
    for (int rover_number = 0; rover_number < teamRover->size(); rover_number++) {
        for (int policy = 0; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
            assert(teamRover->at(rover_number).network_for_agent.at(policy).objective_reward_difference.size() == number_of_objectives);
        }
    }
    
    double temp_sum_value = 0.0;
    for (int poi_number = 0; poi_number < individualPOI->value_poi_vec.size(); poi_number++) {
        temp_sum_value += individualPOI->value_poi_vec.at(poi_number);
    }
    
    
    //Use for testing
    for (int rover_number =0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy_number =0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
            teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_new = teamRover->at(rover_number).network_for_agent.at(policy_number).global_reward_wrt_team - teamRover->at(rover_number).network_for_agent.at(policy_number).local_reward_wrt_team;
        }
    }
    
    for (int rover_number =0 ; rover_number< teamRover->size(); rover_number++) {
        for (int policy_number = 0; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
            if (temp_sum_value <= teamRover->at(rover_number).network_for_agent.at(policy_number).local_reward_wrt_team) {
                FILE* p_miss;
                p_miss =fopen("error_Data_X_Y", "a");
                fprintf(p_miss, "%f \t %f \t %f \t %f \n",temp_sum_value,teamRover->at(rover_number).network_for_agent.at(policy_number).local_reward_wrt_team,teamRover->at(rover_number).network_for_agent.at(policy_number).global_reward_wrt_team,teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team);
                
                for (int x = 0 ; x< teamRover->at(rover_number).network_for_agent.at(policy_number).temp_x.size() ; x++) {
                    fprintf(p_miss,"%f \t %f\n", teamRover->at(rover_number).network_for_agent.at(policy_number).temp_x.at(x),teamRover->at(rover_number).network_for_agent.at(policy_number).temp_y.at(x));
                }
                fclose(p_miss);
            }
        }
    }
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy_number = 0; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
            assert(temp_sum_value > teamRover->at(rover_number).network_for_agent.at(policy_number).local_reward_wrt_team);
            assert(temp_sum_value > teamRover->at(rover_number).network_for_agent.at(policy_number).global_reward_wrt_team);
            assert(temp_sum_value > teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team);
            assert(temp_sum_value > teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_new);
        }
    }
    
}


void select_hall_of_fame(vector<Rover>* teamRover,POI* individualPOI, int number_of_objectives){
    //Makes all hall of fame to false
    for (int rover_number =0 ; rover_number<teamRover->size(); rover_number++) {
        for (int neural_network = 0 ; neural_network < teamRover->at(rover_number).network_for_agent.size(); neural_network++) {
            teamRover->at(rover_number).network_for_agent.at(neural_network).hall_of_fame = false ;
        }
    }
    
    int objective_number = 0;
    assert(objective_number < number_of_objectives);
    
    switch (objective_number ) {
        case 0:
            for (int rover_number =0 ; rover_number < teamRover->size() ; rover_number++) {
                double best_objective = 0;
                int index = 0;
                for (int policy = 0 ; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
                    if (best_objective > teamRover->at(rover_number).network_for_agent.at(policy).objective_reward_difference.at(objective_number)) {
                        best_objective = teamRover->at(rover_number).network_for_agent.at(policy).objective_reward_difference.at(objective_number);
                        index = policy;
                    }
                }
                teamRover->at(rover_number).network_for_agent.at(index).hall_of_fame = true;
            }
            break;
            
        default:
            for (int rover_number = 0; rover_number <teamRover->size(); rover_number++) {
                double temp_best_global = 0.0;
                int index = 0;
                for (int neural_network = 0; neural_network <teamRover->at(rover_number).network_for_agent.size(); neural_network++) {
                    // check for highest value
                    if (temp_best_global < teamRover->at(rover_number).network_for_agent.at(neural_network).difference_reward_wrt_team) {
                        temp_best_global = teamRover->at(rover_number).network_for_agent.at(neural_network).difference_reward_wrt_team;
                        index = neural_network;
                    }
                }
                
                teamRover->at(rover_number).network_for_agent.at(index).hall_of_fame = true;
            }
            break;
    }
    
    
}

void print_to_text(vector<Rover>* teamRover){
//    FILE* pfile;
//    FILE* pfile_1;
//    FILE* pfile_2;
//    FILE* pfile_3;
    FILE* pfile_4;
//    FILE* pfile_5;
//    pfile = fopen("Difference_1","a");
//    pfile_1 = fopen("Difference_2","a");
//    pfile_2 = fopen("global_1", "a");
//    pfile_3 = fopen("gloabl_2","a");
    pfile_4 = fopen("Total", "a");
//    pfile_5 = fopen("Neardistance", "a");
//    for (int rover_number = 0; rover_number <teamRover->size(); rover_number++) {
//        for (int policy = 0; policy < teamRover->at(rover_number).network_for_agent.size(); policy++) {
//            if (rover_number == 0) {
//                fprintf(pfile, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy).difference_reward_wrt_team);
//                fprintf(pfile_2, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy).global_reward_wrt_team);
//            }
//            if (rover_number == 1) {
//                fprintf(pfile_1, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy).difference_reward_wrt_team);
//                fprintf(pfile_3, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy).global_reward_wrt_team);
//            }
//            
//        }
//        if (rover_number == 0) {
//            fprintf(pfile, "\n");
//            fprintf(pfile_2, "\n");
//        }
//        if (rover_number == 1 ) {
//            fprintf(pfile_1, "\n");
//            fprintf(pfile_3, "\n");
//        }
//    }
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy_number = 0; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
            fprintf(pfile_4, "%f \t %f \t %f \n",teamRover->at(rover_number).network_for_agent.at(policy_number).difference_reward_wrt_team,teamRover->at(rover_number).network_for_agent.at(policy_number).global_reward_wrt_team,teamRover->at(rover_number).network_for_agent.at(policy_number).local_reward_wrt_team);
        }
    }
//    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
//        for (int policy_number = 0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
//            for (int closest_distance = 0; closest_distance< teamRover->at(rover_number).network_for_agent.at(policy_number).closest_dist_to_poi.size(); closest_distance++) {
//                fprintf(pfile_5, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy_number).closest_dist_to_poi.at(closest_distance));
//            }
//            fprintf(pfile_5, "\n");
//        }
//        fprintf(pfile_5, "\n");
//    }
    
//    fclose(pfile_5);
    fclose(pfile_4);
//    fclose(pfile);
//    fclose(pfile_1);
//    fclose(pfile_2);
//    fclose(pfile_3);
//    
//    //Remove after testing
//    FILE* pfile_temp;
//    pfile_temp = fopen("closest_distance", "a");
//    
//    for (int rover_number =0 ; rover_number < teamRover->size(); rover_number++) {
//        for (int policy_number =0 ; policy_number < teamRover->at(rover_number).network_for_agent.size(); policy_number++) {
//            for (int distance = 0 ; distance < teamRover->at(rover_number).network_for_agent.at(policy_number).closest_dist_to_poi.size();distance++) {
//                fprintf(pfile_temp, "%f \t",teamRover->at(rover_number).network_for_agent.at(policy_number).closest_dist_to_poi.at(distance));
//            }
//            fprintf(pfile_temp, "\n");
//        }
//        fprintf(pfile_temp, "\n");
//    }
//    fclose(pfile_temp);
//    
}



/***************************
 Main
 **************************/

int main(int argc, const char * argv[]) {
    cout << "Hello, World!\n"<<endl;
    bool VERBOSE = false;
    bool full_verbose = false;
    bool print_text = true;
    srand((unsigned)time(NULL));
    if (test_simulation) {
        test_all_sensors();
        cout<<"All Test"<<endl;
    }
    
//    for (int stat_run =0 ; stat_run < 30; stat_run++) {
        if (run_simulation) {
            if(VERBOSE)
                cout<<"Neural network"<<endl;
            
            //First set up environment
            int number_of_rovers = 1;
            int number_of_poi = 2;
            int number_of_objectives = 1;
            
            //object for environment
            Environment world;
            Environment* p_world = &world;
            
            //Set values of poi's
            POI individualPOI;
            POI* p_poi = &individualPOI;
            
            //Create POI
            
             individualPOI.x_position_poi_vec.push_back(50.0);
             individualPOI.y_position_poi_vec.push_back(100.0);
             individualPOI.x_position_poi_vec.push_back(100.0);
             individualPOI.y_position_poi_vec.push_back(150.0);
//             individualPOI.x_position_poi_vec.push_back(50.0);
//             individualPOI.y_position_poi_vec.push_back(150.0);
//             individualPOI.x_position_poi_vec.push_back(25.0);
//             individualPOI.y_position_poi_vec.push_back(50.0);
//            individualPOI.x_position_poi_vec.push_back(100.0);
//            individualPOI.y_position_poi_vec.push_back(80.0);
//            individualPOI.x_position_poi_vec.push_back(140.0);
//            individualPOI.y_position_poi_vec.push_back(120.0);
//            individualPOI.value_poi_vec.push_back(50.0);
//            individualPOI.value_poi_vec.push_back(50.0);
//            individualPOI.value_poi_vec.push_back(50.0);
//            individualPOI.value_poi_vec.push_back(100.0);
            individualPOI.value_poi_vec.push_back(100.0);
            individualPOI.value_poi_vec.push_back(100.0);
            
            
            //Create 100 poi's 50 at random locations with each group
//            for (int temp_poi = 0 ; temp_poi< 100; temp_poi++) {
//                int temp_x = rand()%100;
//                int temp_y = rand()%100;
//                individualPOI.x_position_poi_vec.push_back(temp_x);
//                individualPOI.y_position_poi_vec.push_back(temp_y);
//                
//                if (temp_poi<50) {
//                    p_world->group_1.push_back(individualPOI);
//                    individualPOI.value_poi_vec.push_back(100);
//                    
//                }else{
//                    p_world->group_2.push_back(individualPOI);
//                    individualPOI.value_poi_vec.push_back(50);
//                }
//            }
//            
//            assert(p_world->group_1.size() == p_world->group_2.size());
            
            
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
            
            
            //check if environment along with rovers are set properly
            assert(individualPOI.x_position_poi_vec.size() == individualPOI.y_position_poi_vec.size());
            assert(individualPOI.value_poi_vec.size() == individualPOI.y_position_poi_vec.size());
            assert(individualPOI.value_poi_vec.size() == number_of_poi);
            assert(teamRover.size() == number_of_rovers);
            if (print_text) {
                FILE* p_location;
                p_location = fopen("locations.txt", "a");
                fprintf(p_location, "Rover Locations \n");
                for (int rover_number =0 ; rover_number < teamRover.size(); rover_number++) {
                    fprintf(p_location, "%f \t %f \n",teamRover.at(rover_number).x_position_vec.at(0),teamRover.at(rover_number).y_position_vec.at(0));
                }
                fprintf(p_location, "POI Location \n");
                for (int poi_location = 0 ; poi_location < individualPOI.value_poi_vec.size(); poi_location++) {
                    fprintf(p_location, "%f \t %f \n",individualPOI.x_position_poi_vec.at(poi_location),individualPOI.y_position_poi_vec.at(poi_location));
                }
                fclose(p_location);
            }
            
            //vector<Population> individual_population_rover; // Individual population is created
            
            
            //Second set up neural networks
            //Create numNN of neural network with pointer
            int numNN = 10;
            vector<unsigned> topology;
            topology.clear();
            topology.push_back(8);
            topology.push_back(10);
            topology.push_back(2);
            
            for (int rover_number =0 ; rover_number < number_of_rovers; rover_number++) {
                teamRover.at(rover_number).create_neural_network_population(numNN, topology);
            }
            
            //First Create teams
            //        set_teams_to_inital(p_rover, numNN);
            //        create_teams(p_rover, numNN);
            
            //        exit(100);
            
            //Generations
            for(int generation =0 ; generation < 300 ;generation++){
                //cout<<"Generation \t \t :::"<<generation<<endl;
                //First Create teams
                set_teams_to_inital(p_rover, numNN);
                create_teams(p_rover, numNN);
                
                for (int rover_number =0; rover_number<teamRover.size(); rover_number++) {
                    teamRover.at(rover_number).random_numbers.clear();      //This is not useful
                    for (int neural_network = 0; neural_network < teamRover.at(rover_number).network_for_agent.size(); neural_network++) {
                        teamRover.at(rover_number).network_for_agent.at(neural_network).closest_dist_to_poi.clear();
                        teamRover.at(rover_number).network_for_agent.at(neural_network).objective_reward_local.clear();
                        teamRover.at(rover_number).network_for_agent.at(neural_network).objective_reward_global.clear();
                        teamRover.at(rover_number).network_for_agent.at(neural_network).objective_reward_difference.clear();
                        teamRover.at(rover_number).network_for_agent.at(neural_network).temp_x.clear();
                        teamRover.at(rover_number).network_for_agent.at(neural_network).temp_y.clear();
                    }
                }
                
//                for (int rover_number = 0; rover_number < teamRover.size(); rover_number++) {
//                    for (int policy_number = 0; policy_number < teamRover.at(rover_number).network_for_agent.size(); policy_number++) {
//                        cout<< teamRover.at(rover_number).network_for_agent.at(policy_number).local_reward_wrt_team<<"\t";
//                    }
//                    cout<<endl;
//                }
                
                //Find scaling number
                double scaling_number = find_scaling_number();
                
                //Setting distance to largest
                for (int network_number =0 ; network_number <numNN; network_number++) {
                    for (int rover_number =0; rover_number < number_of_rovers; rover_number++) {
                        for (int poi_number =0; poi_number<number_of_poi; poi_number++) {
                            teamRover.at(rover_number).network_for_agent.at(network_number).closest_dist_to_poi.push_back(99999999.9999);
                        }
                    }
                }
                //            cout<<"simulation"<<endl;
                for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
                    for (int policy = 0 ; policy < teamRover.at(rover_number).network_for_agent.size(); policy++) {
                        simulation_new_version(p_rover, p_poi, scaling_number, policy, rover_number);
                    }
                }
                
                if (full_verbose) {
                    for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
                        for(int policy = 0 ; policy < p_rover->at(rover_number).network_for_agent.size();policy++){
                            for (int distance = 0 ; distance < p_rover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.size(); distance++) {
                                cout<<p_rover->at(rover_number).network_for_agent.at(policy).closest_dist_to_poi.at(distance)<<"\t";
                            }
                            cout<<endl;
                        }
                        cout<<endl;
                    }
                }
                
                calculate_rewards(p_rover,p_poi,numNN,number_of_objectives);
                //select_hall_of_fame(p_rover, p_poi, number_of_objectives);
                print_to_text(p_rover);
                ccea(p_rover,p_poi,numNN,number_of_objectives);
                
            }
        }
//    }
    
    return 0;
}
