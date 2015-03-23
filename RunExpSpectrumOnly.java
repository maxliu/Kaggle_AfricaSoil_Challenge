import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.AbstractClassifier;

import weka.core.converters.CSVLoader;
import weka.core.logging.Logger;
import weka.core.logging.FileLogger;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.Utils;
import weka.core.Attribute;
import weka.core.Instance;

import weka.Run;

import weka.filters.unsupervised.attribute.Remove;
import weka.filters.Filter;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

import java.io.File;
import java.io.PrintWriter;

class SpecialLogger extends FileLogger {
  
  static void setLogFileName(String fileName) {
    m_Properties.setProperty("LogFile", fileName);
  }
}

public class RunExpSpectrumOnly {
  
  private static FileLogger logger = null;

  private static Instances getTrainingData() throws Exception {

    logger.log(Logger.Level.INFO,"Loading training data");
    CSVLoader csvLoader =  new CSVLoader();
    csvLoader.setSource(new File("training.csv"));
    Instances data = csvLoader.getDataSet();
    return data;
  }

  private static Instances getTestData() throws Exception {

    logger.log(Logger.Level.INFO,"Loading test data");
    CSVLoader csvLoader =  new CSVLoader();
    csvLoader.setSource(new File("sorted_test.csv"));
    Instances data = csvLoader.getDataSet();
    data.insertAttributeAt(new Attribute("Ca"), data.numAttributes());
    data.insertAttributeAt(new Attribute("P"), data.numAttributes());
    data.insertAttributeAt(new Attribute("pH"), data.numAttributes());
    data.insertAttributeAt(new Attribute("SOC"), data.numAttributes());
    data.insertAttributeAt(new Attribute("Sand"), data.numAttributes());
    return data;
  }

  private static Classifier getClassifier(String[] spec) throws Exception {

    String classifierName = spec[0];
    String[] options = new String[spec.length - 1];
    if (options.length > 0) {
      System.arraycopy(spec, 1, options, 0, options.length);
    }
    List<String> prunedMatches = Run.findSchemeMatch(classifierName, false);
    if (prunedMatches.size() > 1) {
      for (String scheme : prunedMatches) {
        logger.log(Logger.Level.INFO,scheme);
      }
      logger.log(Logger.Level.INFO,"More than one scheme name matches -- exiting");
      System.exit(1);
    }
    Classifier classifier = AbstractClassifier.forName(prunedMatches.get(0),
                                                       options);
    
    return classifier;
  }

  private static Classifier[] buildClassifiers(Classifier classifier, Instances data) 
    throws Exception {

    Classifier[] classifiers = new Classifier[5];
    int j = 0;
    for (int i = data.numAttributes() - 5; i < data.numAttributes(); i++) {
      data.setClassIndex(i);
      Classifier myClassifier = AbstractClassifier.makeCopy(classifier);
      FilteredClassifier fc = new FilteredClassifier();
      fc.setClassifier(myClassifier);
      Remove remove = new Remove();
      remove.setAttributeIndices("2-3579," + (i + 1)); 
      remove.setInvertSelection(true);
      fc.setFilter(remove);
      logger.log(Logger.Level.INFO,"Building classifier");
      fc.buildClassifier(data);
      classifiers[j++] = fc;
    }
    return classifiers;
  }

  private static void runCV(Classifier classifier, Instances data) throws Exception {

    data = new Instances(data);
    data.randomize(new Random(1));
    int numFolds = 3;
    for (int i = 0; i < numFolds; i++) {
      Instances trainData = data.trainCV(3, i);
      Classifier[] classifiers = buildClassifiers(classifier, trainData);
      Instances testData = data.testCV(3, i);
      double error = 0;
      for (int k = 0; k < classifiers.length; k++) {
        testData.setClassIndex(testData.numAttributes() - classifiers.length + k);
        double sum = 0;
        for (int j = 0; j < testData.numInstances(); j++) {
          Instance inst = testData.instance(j);
          double p = classifiers[k].classifyInstance(testData.instance(j));
          double diff = p - testData.instance(j).classValue();
          sum += diff * diff;
        }
        error += Math.sqrt(sum / (double)testData.numInstances());
      }
      logger.log(Logger.Level.INFO,"MCRMSE: " + (error / (double) classifiers.length));
    }
  }

  private static void outputPredictions(Classifier[] classifiers, Instances testData, String schemeString) 
    throws Exception {

    PrintWriter pw = new PrintWriter(new File("RESO" + schemeString + ".sub"));

    logger.log(Logger.Level.INFO,"Saving predictions");
    pw.println("PIDN,Ca,P,pH,SOC,Sand");
    for (int i = 0; i < testData.numInstances(); i++) {
      pw.print(testData.instance(i).stringValue(0) + ",");
      for (int j = 0; j < classifiers.length; j++) {
        testData.setClassIndex(testData.numAttributes() - classifiers.length + j);
        pw.print(classifiers[j].classifyInstance(testData.instance(i)));
        if (j < classifiers.length - 1) {
          pw.print(",");
        }
      }
      pw.println();
    }
    pw.close();
  }


  public static void main(String[] args) throws Exception {
    
    String schemeString = new String(args[0]);
    for (int i = 1; i < args.length; i++) {
      schemeString += "_" + args[i];
    }
    SpecialLogger.setLogFileName("RESO" + schemeString + ".log");
    logger = new SpecialLogger();
    WekaPackageManager.loadPackages(false, true, false);
    Classifier classifier = getClassifier(args);
    Instances trainData = getTrainingData();
    //runCV(classifier, trainData);
    outputPredictions(buildClassifiers(classifier, trainData), getTestData(), schemeString);
    logger.log(Logger.Level.INFO,"Finished");
  }
}