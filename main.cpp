#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/vfh.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>

using namespace std;

int main(int argc, char *argv[])
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the cloud data
  pcl::PCDReader reader;
  reader.read("pcl_power_supply.pcd", *cloud);

  // Create the filtering object
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.3, 0.8);
  //pass.setFilterLimitsNegative (true);
  pass.filter (*cloud_filtered);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

  //Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::ExtractIndices<pcl::PointXYZ> extract;

  //optional
  seg.setOptimizeCoefficients(true);

  //Mantdatory
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);

  seg.setInputCloud(cloud_filtered);
  seg.segment(*inliers, *coefficients);

  //Extract the planar inliers from the input cloud
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(inliers);
  extract.setNegative(true);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
  extract.filter(*cloud_plane);
  cerr<< "PointCloud representing the planar component: " << cloud_plane->points.size() << "data points." << endl;

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

  //Estimate the normals
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cloud_plane);

  //Create an empty kdtree representation, and pass it to the FPFH estimation object
  //Its content will be filled inside the object, based on the given input dataset (as no other search surface is given)
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  ne.setSearchMethod(tree);
  pcl::PointCloud<pcl::Normal>& cloud_normals = *normals;

  //Use all neighbors in a sphere of radius 3cm
  ne.setRadiusSearch(0.03);
  ne.compute(cloud_normals);

  //create the VFH estimation class, and pass the input dataset+normals to it
  pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
  vfh.setInputCloud(cloud_plane);
  vfh.setInputNormals(normals);

  //Create an empty kdtree representation, and pass it to the FPFH estimation objects.
  //Its content will be filled inside the object, based on the given input dataset (as no other search surface is given)
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>());
  vfh.setSearchMethod(tree2);

  //Output the dataset
  pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs(new pcl::PointCloud<pcl::VFHSignature308>());

  //Compute the features of objects
  vfh.compute(*vfhs);

  //Plot the histogram to visualize it
  pcl::visualization::PCLPlotter *plotter = new pcl::visualization::PCLPlotter("VFH Descriptor");
  plotter->setShowLegend(true);
  plotter->setTitle("VFH Descriptor");
  plotter->addFeatureHistogram(*vfhs, 308);
  plotter->plot();

  return (0);
}
