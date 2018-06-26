/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// set the number of particles to draw
	num_particles = 50;

	// create a normal (Gaussian) distribution for x, y and theta
	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	for (int i = 0; i < num_particles; ++i) {
		// Create a new particle
		Particle p;

		// sample from these normal distrubtions
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		// push the new particle to the particle vector
		particles.push_back(p);
		// cout << "x, y, theta = " << p.x << ", " << p.y << ", " << p.theta << endl;
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// create a normal (Gaussian) distribution for distance x, y, and theta
	default_random_engine gen;

	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	for (int i = 0; i < num_particles; ++i) {

		// make prediction and add noise from normal distrubtions
		particles[i].x     += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
		particles[i].y     += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
		particles[i].theta += yaw_rate * delta_t + dist_theta(gen);

		particles[i].weight = 1.0;

		// cout << "x, y, theta = " << particles[i].x << ", " << particles[i].y << ", " << particles[i].theta << endl;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	//// ! Since the following term is a constant for every particles, 
	//// ! there's no need to multiple by this term in the probability calculation 
	//// ! to save some computing time.

    // calculate normalization term
    // double gauss_norm = 1./ (2 * M_PI * std_landmark[0] * std_landmark[0]);

    for (unsigned int i = 0; i < num_particles; ++i) {

    	double x_part = particles[i].x;
    	double y_part = particles[i].y;
    	double theta_part = particles[i].theta;

	    // define landmark prediction vector to store sensible landmarks:
	    Map map_prediction = map_landmarks;

	    // loop over all landmarks and estimate the prediction in the range:
        for (unsigned int l = 0; l < map_prediction.landmark_list.size(); ++l) {

            // estimate pseudo range for each single landmark 
            double x_l = map_prediction.landmark_list[l].x_f;
            double y_l = map_prediction.landmark_list[l].y_f;

            double range_l = dist(x_l, y_l, x_part, y_part);
            
            //check if distances are in the detection rage: 
            if (range_l > sensor_range) {
                map_prediction.landmark_list.erase(map_prediction.landmark_list.begin() + l);
            }
        }

    	// loop over all ovservations that have been transformed
    	// from VEHICLE'S coordinate system to MAP'S coordinate system
    	for (unsigned int j = 0; j < observations.size(); ++j) {

    		double x_obs = observations[j].x;
    		double y_obs = observations[j].y;

    		double x_m = x_part + cos(theta_part) * x_obs - sin(theta_part) * y_obs;
    		double y_m = y_part + sin(theta_part) * x_obs + cos(theta_part) * y_obs;

			// particles[i].sense_x.push_back(x_m);
			// particles[i].sense_y.push_back(y_m);

	        // find the minimum distance from the predicted to observed landmark
    		// and assign it to the particle association
	        double dist_min = std::numeric_limits<const double>::infinity();
	        double x_mu;
	        double y_mu;

	        for (unsigned int  k = 0; k < map_prediction.landmark_list.size(); ++k) {

	        	double range = dist(x_m, y_m, map_prediction.landmark_list[k].x_f, map_prediction.landmark_list[k].y_f);

	        	if (range < dist_min) {
	        		dist_min = range;
	        		x_mu = map_prediction.landmark_list[k].x_f;
	        		y_mu = map_prediction.landmark_list[k].y_f;
	        		// cout << "  k = " << k << " dist min: " << dist_min << endl;
	        	}
	        }

			// calculate exponent
			double exponent = (x_m - x_mu)*(x_m - x_mu)/(2 * std_landmark[0]*std_landmark[0]) + (y_m - y_mu)*(y_m - y_mu)/(2 * std_landmark[1]*std_landmark[1]);

			// calculate weight using exponent 
			// ignore normalization terms
			particles[i].weight *= exp(-exponent);

			// cout << " j = " << j << " " << exp(-exponent) << endl;

    	}

	// cout << "i = " << i << " " << particles[i].weight << endl;

    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// setup the random bits
	default_random_engine gen;

	std::vector<double> weights;
	for(unsigned int i = 0; i< num_particles; ++i)
		weights.push_back(particles[i].weight);
    std::discrete_distribution<> d(weights.begin(), weights.end());

    // resample with replacement from the existing particle based on their weights
	std::vector<Particle> particles_resampled;

    for(unsigned int n = 0; n < num_particles; ++n) {
        particles_resampled.push_back(particles[d(gen)]);
    	// particles_resampled[n].weight = 1.0; 
    }
    particles = particles_resampled;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
